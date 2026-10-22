"""
test_lbm_propagation.py — coupled_step must actually stream populations.

Collision conserves mass cell by cell, so a density bump in a molten fluid
box can only leave its cell through streaming. A step that discards the
streamed buffer leaves the bump in place indefinitely.
"""

from __future__ import annotations

import sys

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin


def run(n_steps: int = 6, max_left: float = 0.5) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=16, ny=16, nz=16, dx=3e-4, max_tracers=10)
    twin.enable_heat_loss = False
    twin.enable_lorentz = False
    twin.reset(T_ambient=float(twin.mat.T_liquidus) + 200.0, test_fluid_domain=True)
    g = twin.grid
    c = (8, 8, 8)
    f = g.f_src.to_numpy()
    rho0 = float(f[:, c[0], c[1], c[2]].sum())
    f[:, c[0], c[1], c[2]] *= 1.2
    g.f_src.from_numpy(f)
    for _ in range(n_steps):
        twin.step(0.0, 0.0, is_welding=False)
    rho_c = float(g.f_src.to_numpy()[:, c[0], c[1], c[2]].sum())
    left = (rho_c - rho0) / (0.2 * rho0)
    print(f"[lbm_propagation] density bump left in its cell after {n_steps} steps: {left:.3f}")
    if not left < max_left:
        raise AssertionError(
            f"density bump did not propagate ({left:.3f} of it left, need < {max_left})"
        )
    return left


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
