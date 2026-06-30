"""
test_mass_conservation.py — LBM mass drift after stream+collision steps.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def run(n_steps: int = 500, threshold_pct: float = 2.0) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=32, ny=32, nz=16, dx=3e-4, C_darcy=0.0, max_tracers=50)
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    rho0 = float(g.rho.to_numpy().mean())

    for _ in range(n_steps):
        kernels.clear_forces(g.Fx, g.Fy, g.Fz)
        kernels.collide_srt(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.tau, twin.omega, 1.0,
            0.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        kernels.stream(
            g.f_dst, g.f_src,
            g.flags, g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        g.swap_buffers()

    rho1 = float(g.rho.to_numpy().mean())
    drift_pct = abs(rho1 - rho0) / rho0 * 100.0
    print(f"[mass_conservation] rho drift = {drift_pct:.4f}%  (threshold {threshold_pct}%)")
    if drift_pct >= threshold_pct:
        raise AssertionError(f"Mass drift {drift_pct:.4f}% >= {threshold_pct}%")
    return drift_pct


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
