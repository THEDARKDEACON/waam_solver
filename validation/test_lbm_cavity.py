"""
test_lbm_cavity.py — Forced cavity flow smoke (optional Phase 0+).

Uniform body force drives Poiseuille-like flow in a fluid-filled channel.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def run(n_steps: int = 150, min_u: float = 1e-6) -> float:
    init_taichi(backend="cpu")
    nx, ny, nz = 64, 32, 6
    tau_test = 1.0
    omega_test = 1.0 / tau_test
    nu_lu = (tau_test - 0.5) / 3.0
    fx_lu = 1e-4

    twin = WAAMTwin(nx=nx, ny=ny, nz=nz, dx=3e-4, C_darcy=0.0, max_tracers=10)
    g = twin.grid
    rho0 = 1.0

    kernels.init_poiseuille_channel(
        g.f_a, g.rho, g.ux, g.uy, g.uz,
        g.f_l, g.phi, g.flags,
        rho0, g.ny, g.nz, fx_lu, nu_lu,
        g.FLAG_FLUID, g.FLAG_SOLID,
    )
    kernels.init_aux_fields(
        g.T_max, g.T_prev, g.dT_dt, g.Fx, g.Fy, g.Fz, twin.T_amb,
        g.porosity_active, g.tracer_head, g.max_tracers,
    )
    kernels.stream(g.f_a, g.f_b, g.flags, g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz)

    for _ in range(n_steps):
        kernels.set_uniform_Fx(g.Fx, g.Fy, g.Fz, fx_lu, g.flags, g.FLAG_FLUID)
        kernels.collide_srt(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            tau_test, omega_test, 1.0,
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

    ux_np = g.ux.to_numpy()
    flags_np = g.flags.to_numpy()
    fluid = flags_np == g.FLAG_FLUID
    u_mean = float(np.abs(ux_np[fluid]).mean()) if fluid.any() else 0.0

    print(f"[lbm_cavity] mean |ux| fluid = {u_mean:.3e}  (min {min_u:.1e})")
    if u_mean < min_u:
        raise AssertionError(f"Cavity flow too weak: {u_mean:.3e}")
    return u_mean


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
