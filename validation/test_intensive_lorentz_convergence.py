"""
test_intensive_lorentz_convergence.py — Warm-start Jacobi residual under load.

Seeds a large molten patch, solves Lorentz repeatedly, and asserts the
relative residual drops below tolerance within the configured iteration budget
after warm-start (second solve).
"""

from __future__ import annotations

import os
import sys

from waam_twin import WAAMTwin
from waam_twin.physics import forces, weld_forces
from waam_twin.runtime import init_taichi


def run() -> None:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(
        nx=40, ny=28, nz=28, dx=3.5e-4,
        enable_vof=True,
        enable_lorentz=True,
        welding_current_A=200.0,
        lorentz_jacobi_iters=250,
        max_tracers=20,
    )
    twin.lorentz_jacobi_tol = 5.0e-4
    twin.wire_feed_m_s = 5.0 / 60.0
    twin.reset()
    g = twin.grid
    i_arc = g.nx // 3
    j = g.ny // 2
    k_arc = twin.nz_solid + 1

    for di in range(-6, 7):
        for dj in range(-5, 6):
            for dk in range(0, 5):
                ii, jj, kk = i_arc + di, j + dj, k_arc + dk
                if 0 <= ii < g.nx and 0 <= jj < g.ny and 0 <= kk < g.nz:
                    g.f_l[ii, jj, kk] = 0.95
                    g.T[ii, jj, kk] = twin.mat.T_liquidus + 400.0
                    g.flags[ii, jj, kk] = g.FLAG_FLUID
                    g.phi[ii, jj, kk] = 1.0

    # Cold start
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    twin._lorentz_unconverged = 0
    weld_forces.solve_lorentz(twin, g, float(i_arc), float(j), float(k_arc))
    cold_fail = int(getattr(twin, "_lorentz_unconverged", 0) or 0)

    # Warm starts (potential already filled)
    warm_fails = 0
    for _ in range(8):
        forces.clear_forces(g.Fx, g.Fy, g.Fz)
        before = int(getattr(twin, "_lorentz_unconverged", 0) or 0)
        weld_forces.solve_lorentz(twin, g, float(i_arc), float(j), float(k_arc))
        after = int(getattr(twin, "_lorentz_unconverged", 0) or 0)
        if after > before:
            warm_fails += 1

    fx = g.Fx.to_numpy()
    fz = g.Fz.to_numpy()
    fl = g.f_l.to_numpy()
    mask = fl > 0.5
    fmax = float((fx[mask] ** 2 + fz[mask] ** 2).max() ** 0.5) if mask.any() else 0.0

    print(
        f"[intensive_lorentz] cold_unc={cold_fail}  warm_fail_steps={warm_fails}/8  "
        f"|F|_max={fmax:.3e}  iters={twin.lorentz_jacobi_iters}  "
        f"tol={twin.lorentz_jacobi_tol}"
    )
    if fmax <= 0.0:
        raise AssertionError("Lorentz produced no force on molten patch")
    if warm_fails > 2:
        raise AssertionError(
            f"Lorentz warm-start failed to converge on {warm_fails}/8 solves — "
            f"raise lorentz_jacobi_iters"
        )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
