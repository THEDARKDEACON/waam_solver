"""
test_evaporative_cooling.py — Free-surface evaporative sink removes enthalpy.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=24, ny=16, nz=18, dx=3.5e-4,
        enable_vof=True,
        enable_enthalpy_cap=True,
        enable_evaporative_cooling=True,
        T_vapor_cap_K=3200.0,
        heat_source="goldak",
        arc_power_W=1500.0,
        arc_efficiency=0.8,
        max_tracers=10,
    )
    twin.evap_cooling_scale = 50.0
    twin.reset()
    g = twin.grid

    # Seed a hot free-surface patch above boiling.
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    i0, j0 = g.nx // 2, g.ny // 2
    k0 = max(1, twin.nz_solid)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            i, j, k = i0 + di, j0 + dj, k0
            if not (0 <= i < g.nx and 0 <= j < g.ny):
                continue
            flags[i, j, k] = g.FLAG_FLUID
            phi[i, j, k] = 0.7
            fl[i, j, k] = 1.0
            T[i, j, k] = 3150.0
            H[i, j, k] = twin.H_liq + twin.cp_rho * (3150.0 - twin.mat.T_liquidus)
            if k + 1 < g.nz:
                flags[i, j, k + 1] = g.FLAG_GAS
                phi[i, j, k + 1] = 0.0
    g.T.from_numpy(T)
    g.H.from_numpy(H)
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)

    H0 = float(g.H.to_numpy().sum()) * (g.dx ** 3)
    # One idle step still runs evaporative sink in coupled_step.
    twin.step(i0 * g.dx, j0 * g.dx, is_welding=False)
    H1 = float(g.H.to_numpy().sum()) * (g.dx ** 3)
    removed = H0 - H1
    telem = twin.get_telemetry()
    print(
        f"[evaporative_cooling] dH={removed:.4e}J  "
        f"telem_step={telem.get('evap_energy_J_step', 0):.4e}J  "
        f"scale={twin.evap_cooling_scale}"
    )
    if removed <= 0.0 and float(telem.get("evap_energy_J_step", 0.0)) <= 0.0:
        raise AssertionError("Evaporative sink did not remove enthalpy")
    return removed


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
