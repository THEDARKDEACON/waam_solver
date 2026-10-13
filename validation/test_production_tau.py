"""Phase 9 — production-regime operator smoke at τ≈0.5007."""

from __future__ import annotations

import os


def run() -> None:
    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.units import LatticeScale

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    # Production-like dx/dt → τ near 0.5
    dx = 4e-4
    nu = 6e-7
    dt = 4e-5
    scale = LatticeScale(dx=dx, dt=dt, rho=7800.0)
    tau = scale.tau_from_nu(nu)
    assert 0.500 < tau < 0.52, f"unexpected tau={tau}"

    twin = WAAMTwin(nx=24, ny=16, nz=16, dx=dx, max_tracers=8, C_darcy=0.0)
    twin.grid.dt = dt
    twin.grid.tau = tau
    twin.enable_vof = False
    twin.enable_marangoni = False
    twin.enable_lorentz = False
    twin.reset()
    for _ in range(20):
        twin.step(0.5 * twin.grid.nx * dx, 0.5 * twin.grid.ny * dx, is_welding=False)
    # No NaNs in rho
    import numpy as np
    rho = twin.grid.rho.to_numpy()
    assert np.isfinite(rho).all()
    print(f"[production_tau] tau={tau:.6f}  rho ok after 20 steps")


if __name__ == "__main__":
    run()
    print("PASS")
