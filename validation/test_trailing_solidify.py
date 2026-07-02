"""
test_trailing_solidify.py — Trailing liquid behind the torch is clamped back to solid.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin import kernels


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=24, ny=12, nz=14, dx=3e-4, enable_bead_freeze=True)
    twin.reset()
    g = twin.grid
    i, j, k = 6, g.ny // 2, twin.nz_solid + 1
    g.T[i, j, k] = twin.mat.T_liquidus + 20.0
    g.f_l[i, j, k] = 1.0
    g.phi[i, j, k] = 1.0
    g.flags[i, j, k] = g.FLAG_FLUID
    g.ux[i, j, k] = 0.04
    g.H[i, j, k] = twin.cp_rho * (twin.mat.T_liquidus + 40.0) + twin.L_rho

    kernels.solidify_trailing_pool_scalar(
        g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz, twin.cp_rho,
        14.0, float(j), float(k), 1.0, 0.0, 0.0,
        4.0, twin.mat.T_liquidus + 35.0, twin.mat.T_solidus,
        g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
    )
    assert int(g.flags[i, j, k]) == g.FLAG_SOLID
    assert float(g.f_l[i, j, k]) == 0.0
    assert float(g.ux[i, j, k]) == 0.0
    print("[trailing_solidify] trailing pool solidified OK")


if __name__ == "__main__":
    run()
    print("PASS")
