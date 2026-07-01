"""
test_hydrostatic_gravity.py — Hydrostatic body force on liquid cells.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import forces


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=24, ny=16, nz=18, dx=3e-4, enable_hydrostatic_gravity=True)
    twin.reset()
    g = twin.grid
    i, j, k = g.nx // 2, g.ny // 2, twin.nz_solid + 2
    g.f_l[i, j, k] = 1.0
    g.flags[i, j, k] = g.FLAG_FLUID
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.add_hydrostatic_gravity(
        g.Fz, g.f_l, g.flags, g.mat.rho, twin.g_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    fz = float(g.Fz[i, j, k])
    assert fz < 0.0, "hydrostatic gravity should pull liquid downward (+z up)"
    print(f"[hydrostatic] Fz={fz:.3e} lu")


if __name__ == "__main__":
    run()
    print("PASS")
