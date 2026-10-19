"""Enthalpy inversion and the 1520 C contour.

Host checks do not need a GPU. ``run()`` also recovers temperature on a
tiny CPU twin when Quadrants is importable.
"""

from __future__ import annotations

import numpy as np

from waam_twin.material_tables import sensible_enthalpy_J_kg, volumetric_enthalpy_J_m3
from waam_twin.materials import load_material
from waam_twin.measure.geometry import fusion_contour_mm


def _invert_specific(target: float, knots, fallback: float, T_hi: float = 4000.0) -> float:
    lo, hi = 0.0, float(T_hi)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if sensible_enthalpy_J_kg(mid, knots, fallback) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_constant_cp_matches_the_old_anchor():
    rho, cp, L = 7850.0, 650.0, 260000.0
    Ts, Tl = 1753.15, 1793.15
    for T in (293.0, Ts, 400.0):
        H = volumetric_enthalpy_J_m3(rho, cp, [], T, Ts, Tl, L)
        assert abs(H - rho * cp * T) < 1.0
    H_liq = volumetric_enthalpy_J_m3(rho, cp, [], Tl, Ts, Tl, L)
    assert abs(H_liq - (rho * cp * Tl + rho * L)) < 1.0


def test_mild_steel_hump_inverts():
    mat = load_material("materials/validated/MildSteel.ansys.v1.yaml")
    knots = list(mat.tables.cp)
    T = 735.0 + 273.15
    specific = sensible_enthalpy_J_kg(T, knots, mat.cp)
    straight = specific / mat.cp_at(T)
    assert abs(straight - (T - 273.15)) > 10.0
    back = _invert_specific(specific, knots, mat.cp)
    assert abs(back - T) < 0.05


def test_fusion_contour_uses_cell_centers():
    dx = 0.0004
    nz_solid = 10
    shape = (5, 3, 12)
    metal = np.zeros(shape, dtype=bool)
    metal[:, :, :nz_solid] = True

    depth = np.full(shape, 300.0)
    depth[2, 1, 7] = 1000.0
    depth[2, 1, 8] = 2000.0
    d = fusion_contour_mm(depth, metal, 1500.0, dx, nz_solid, 0.0, 1.0)
    assert abs(d["pool_depth_mm"] - 0.8) < 1e-6

    width = np.full(shape, 300.0)
    width[0, 1, 8] = 1400.0
    width[1, 1, 8] = 1600.0
    width[2, 1, 8] = 1400.0
    w = fusion_contour_mm(width, metal, 1520.0, dx, nz_solid, 0.0, 1.0)
    assert abs(w["pool_width_mm"] - 0.32) < 1e-6


def run() -> None:
    test_constant_cp_matches_the_old_anchor()
    test_mild_steel_hump_inverts()
    test_fusion_contour_uses_cell_centers()

    from waam_twin import WAAMTwin
    from waam_twin.runtime import init_taichi, reset_taichi
    from waam_twin.solvers.coupled_step import _update_phase
    from waam_twin.twin import _enthalpy_J_m3

    reset_taichi()
    init_taichi(backend="cpu")
    plate = load_material("materials/validated/MildSteel.ansys.v1.yaml")
    wire = load_material("materials/placeholders/ER70S-6.yaml")
    twin = WAAMTwin(
        material=wire,
        plate_material=plate,
        nx=8,
        ny=8,
        nz=12,
        dx=4e-4,
        max_tracers=4,
    )
    twin.reset()
    g = twin.grid
    H = g.H.to_numpy()
    flags = g.flags.to_numpy()
    frac = g.alloy_frac.to_numpy()
    ip, jp = 4, 4
    kp = max(0, twin.nz_solid - 2)
    if abs(float(frac[ip, jp, kp]) - 1.0) > 1e-6:
        raise AssertionError("plate cell should be alloy_frac 1")
    T_hot = 735.0 + 273.15
    H[ip, jp, kp] = _enthalpy_J_m3(plate, T_hot)
    kw = min(g.nz - 2, twin.nz_solid + 1)
    flags[ip, jp, kw] = g.FLAG_FLUID
    frac[ip, jp, kw] = 0.0
    H[ip, jp, kw] = wire.rho * wire.cp * 400.0
    g.H.from_numpy(H)
    g.flags.from_numpy(flags)
    g.alloy_frac.from_numpy(frac)
    _update_phase(twin, g)
    T = g.T.to_numpy()
    if abs(float(T[ip, jp, kp]) - T_hot) > 1.0:
        raise AssertionError(f"plate 735 C recovered as {float(T[ip, jp, kp]) - 273.15:.2f} C")
    if abs(float(T[ip, jp, kw]) - 400.0) > 0.5:
        raise AssertionError(f"constant-cp cell recovered as {float(T[ip, jp, kw]):.2f} K")
    print("enthalpy curve and fusion contour ok")


if __name__ == "__main__":
    run()
