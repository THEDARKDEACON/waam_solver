"""
test_intensive_vapor_physics.py — Soft-onset recoil + evaporative sink under heat.

Drives a surface into the onset→boil band and asserts:
  - evaporative enthalpy removal accumulates
  - soft-onset CC recoil is nonzero below T_boil
  - above T_boil, |F_recoil| increases
  - vapor cap does not dominate (n_cap stays low after cooling)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from waam_twin import WAAMTwin, kernels
from waam_twin.physics import weld_forces
from waam_twin.platform import init_taichi


def _seed_hot_surface(twin: WAAMTwin, T_hot: float) -> tuple[int, int, int]:
    g = twin.grid
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    i0, j0 = g.nx // 2, g.ny // 2
    k0 = max(1, twin.nz_solid)
    for di in range(-3, 4):
        for dj in range(-3, 4):
            i, j, k = i0 + di, j0 + dj, k0
            if not (0 <= i < g.nx and 0 <= j < g.ny):
                continue
            flags[i, j, k] = g.FLAG_FLUID
            phi[i, j, k] = 0.65
            fl[i, j, k] = 1.0
            T[i, j, k] = T_hot
            H[i, j, k] = twin.H_liq + twin.cp_rho * (T_hot - twin.mat.T_liquidus)
            if k + 1 < g.nz:
                flags[i, j, k + 1] = g.FLAG_GAS
                phi[i, j, k + 1] = 0.05
                # Vertical φ gradient for recoil interface detection
                phi[i, j, k] = 0.7
    g.T.from_numpy(T)
    g.H.from_numpy(H)
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)
    return i0, j0, k0


def run() -> None:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(
        nx=36, ny=24, nz=24, dx=3.5e-4,
        enable_vof=True,
        enable_recoil=True,
        use_recoil_clausius_clapeyron=True,
        enable_evaporative_cooling=True,
        enable_enthalpy_cap=True,
        T_vapor_cap_K=3200.0,
        T_boiling_K=3000.0,
        heat_source="goldak",
        arc_power_W=2800.0,
        arc_efficiency=0.72,
        max_tracers=20,
    )
    twin.recoil_accommodation = 0.25
    twin.evap_cooling_scale = 25.0
    twin.reset()

    T_boil = twin.T_boiling_K
    T_onset = weld_forces.recoil_onset_K(twin)
    T_soft = 0.5 * (T_onset + T_boil)
    i0, j0, k0 = _seed_hot_surface(twin, T_soft)
    g = twin.grid

    # Soft-onset recoil magnitude
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    weld_forces.apply_recoil(twin, g, float(i0), float(j0), float(k0))
    F_soft = float(np.abs(g.Fz.to_numpy()).max())

    # Above boil — stronger
    _seed_hot_surface(twin, T_boil + 250.0)
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    weld_forces.apply_recoil(twin, g, float(i0), float(j0), float(k0))
    F_hot = float(np.abs(g.Fz.to_numpy()).max())

    # Evaporative removal over several idle steps
    _seed_hot_surface(twin, T_boil + 100.0)
    H0 = float(g.H.to_numpy().sum()) * (g.dx ** 3)
    evap_cum = 0.0
    for _ in range(40):
        twin.step(i0 * g.dx, j0 * g.dx, is_welding=False)
        evap_cum += float(twin.get_telemetry().get("evap_energy_J_step", 0.0))
    H1 = float(g.H.to_numpy().sum()) * (g.dx ** 3)
    telem = twin.get_telemetry()

    print(
        f"[intensive_vapor] T_onset={T_onset:.0f} T_boil={T_boil:.0f}  "
        f"|F|_soft={F_soft:.3e} |F|_hot={F_hot:.3e}  "
        f"dH={H0 - H1:.3e}J  evap_cum={evap_cum:.3e}J  "
        f"n_cap={telem.get('n_cells_at_vapor_cap', 0)}"
    )
    if F_soft <= 0.0:
        raise AssertionError("soft-onset recoil inactive in onset→boil band")
    if F_hot <= F_soft:
        raise AssertionError("recoil above boil should exceed soft-onset magnitude")
    if (H0 - H1) <= 0.0 and evap_cum <= 0.0:
        raise AssertionError("evaporative sink removed no enthalpy over intensive window")
    if int(telem.get("n_cells_at_vapor_cap", 0)) > 50:
        raise AssertionError("too many vapor-cap cells — evaporative cooling ineffective")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
