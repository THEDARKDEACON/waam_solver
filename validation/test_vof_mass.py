"""
test_vof_mass.py — VOF metal volume conservation under uniform advection.

A metal slab (φ = 1 inside, 0 outside, all cells FLUID so the velocity acts
everywhere) is advected by a uniform u_x. The slab stays clear of the walls,
so the donor-cell flux form must conserve Σφ to round-off and the slab
centroid must move by u·t.

The earlier version filled the whole closed box with φ = 1 and pushed it
into the east wall. Uniform flow into a no-flux wall is not divergence-free,
so φ piled above 1 at the wall and the [0, 1] clamp deleted it (5 % drift
over 80 steps = 1.6 cells / 32). That measured the test setup, not the
advection scheme; it passed only while x was periodic.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels

U_X = 0.02
SLAB = (8, 16)  # x-range of the initial slab, cells


def run(n_steps: int = 80, threshold_pct: float = 0.01) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=32, ny=32, nz=16, dx=3e-4, C_darcy=0.0, max_tracers=10, enable_vof=True)
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    flags_np = g.flags.to_numpy()
    fluid = flags_np == g.FLAG_FLUID
    phi_np = np.zeros(flags_np.shape, dtype=np.float32)
    phi_np[SLAB[0]:SLAB[1]] = 1.0
    phi_np[flags_np == g.FLAG_SOLID] = 1.0
    phi_np[~fluid & (flags_np != g.FLAG_SOLID)] = 0.0
    g.phi.from_numpy(phi_np)

    ux_np = np.zeros(flags_np.shape, dtype=np.float32)
    ux_np[fluid] = U_X
    g.ux.from_numpy(ux_np)
    g.uy.fill(0.0)
    g.uz.fill(0.0)

    xs = (np.arange(g.nx) + 0.5)[:, None, None]

    def metal_stats() -> tuple[float, float]:
        p = g.phi.to_numpy()
        m = np.where(fluid, p, 0.0)
        tot = float(m.sum())
        return tot, float((m * xs).sum() / max(tot, 1e-30))

    m0, x0 = metal_stats()
    for _ in range(n_steps):
        kernels.advect_phi(
            g.phi_tmp, g.phi, g.ux, g.uy, g.uz, g.flags,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        g.phi.copy_from(g.phi_tmp)
        kernels.reinitialize_phi(g.phi, g.flags, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_FLUID)
    m1, x1 = metal_stats()

    drift = abs(m1 - m0) / max(m0, 1e-9) * 100.0
    shift = x1 - x0
    expected = U_X * n_steps
    print(
        f"[vof_mass] metal drift = {drift:.4f}%  (threshold {threshold_pct}%)  "
        f"centroid shift {shift:.3f} cells (expected {expected:.3f})"
    )
    if drift >= threshold_pct:
        raise AssertionError(f"VOF mass drift {drift:.4f}% >= {threshold_pct}%")
    if abs(shift - expected) > 0.05 * expected:
        raise AssertionError(
            f"slab centroid moved {shift:.3f} cells, expected {expected:.3f} (advection inactive?)"
        )
    return drift


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
