"""
test_lorentz_physical_scale.py — |J×B|/ρ within order-of-magnitude of SI reference.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.physics import forces, weld_forces
from waam_twin.physics.lorentz_physics import lorentz_reference_accel_m_s2
from waam_twin.platform import init_taichi


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=32, ny=20, nz=22, dx=3e-4,
        enable_vof=True, enable_lorentz=True,
        welding_current_A=180.0,
        lorentz_jacobi_iters=60,
        max_tracers=10,
    )
    twin.wire_feed_m_s = 8.0 / 60.0
    twin.reset()
    g = twin.grid
    i_arc = g.nx // 3
    j = g.ny // 2
    k_arc = twin.nz_solid + 1

    for di in range(-4, 5):
        for dj in range(-4, 5):
            for dk in range(0, 4):
                ii, jj, kk = i_arc + di, j + dj, k_arc + dk
                if 0 <= ii < g.nx and 0 <= jj < g.ny and 0 <= kk < g.nz:
                    g.f_l[ii, jj, kk] = 0.95
                    g.T[ii, jj, kk] = twin.mat.T_liquidus + 300.0
                    g.flags[ii, jj, kk] = g.FLAG_FLUID
                    g.phi[ii, jj, kk] = 1.0

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    weld_forces.solve_lorentz(twin, g, float(i_arc), float(j), float(k_arc))

    fx = g.Fx.to_numpy()
    fy = g.Fy.to_numpy()
    fz = g.Fz.to_numpy()
    fl = g.f_l.to_numpy()
    mask = fl > 0.5
    if not mask.any():
        raise AssertionError("no liquid patch for Lorentz test")

    # Lattice force → physical acceleration [m/s²]
    a_lu = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    a_phys = a_lu[mask].max() * g.dx / (g.dt ** 2)
    r_pool = twin.sigma_cells * g.dx
    a_ref = lorentz_reference_accel_m_s2(twin.welding_current_A, r_pool, twin.mat.rho)
    ratio = a_phys / max(a_ref, 1e-9)

    print(f"[lorentz_physical] a_sim={a_phys:.2f} m/s²  a_ref={a_ref:.2f} m/s²  ratio={ratio:.3f}")
    if ratio < 0.05 or ratio > 20.0:
        raise AssertionError(
            f"Lorentz acceleration ratio {ratio:.3f} outside [0.05, 20] — check SI EM scaling"
        )


if __name__ == "__main__":
    run()
    print("PASS")
