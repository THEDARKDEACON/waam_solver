"""
test_bead_freeze.py — Solidified metal has zero velocity when bead freeze enabled.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import free_surface


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=24, ny=14, nz=18, dx=3e-4, enable_bead_freeze=True)
    twin.reset()
    g = twin.grid
    i, j, k = g.nx // 2, g.ny // 2, twin.nz_solid
    g.T[i, j, k] = 500.0
    g.f_l[i, j, k] = 0.0
    g.phi[i, j, k] = 1.0
    g.flags[i, j, k] = g.FLAG_FLUID
    g.ux[i, j, k] = 0.05
    g.uy[i, j, k] = -0.02
    g.uz[i, j, k] = 0.03

    free_surface.solidify_cooled_metal(
        g.T, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz,
        twin.mat.T_solidus, True,
        g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
    )
    assert int(g.flags[i, j, k]) == g.FLAG_SOLID
    assert float(g.ux[i, j, k]) == 0.0
    assert float(g.uy[i, j, k]) == 0.0
    assert float(g.uz[i, j, k]) == 0.0
    print("[bead_freeze] SOLID velocity zeroed OK")


if __name__ == "__main__":
    run()
    print("PASS")
