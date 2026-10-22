"""
test_lbm_poiseuille.py — Steady force-driven duct flow under Guo forcing.

init_poiseuille_channel walls off y and z, so the channel is a rectangular
duct; the reference is the duct series solution with half-way bounce-back
walls, sampled across y on the mid-z line after the flow reaches steady state.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def _duct_profile(js, k, ny, nz, terms: int = 41) -> np.ndarray:
    """Unnormalised duct velocity at cells (js, k); walls half-way outside the fluid nodes."""
    a = (ny - 2) / 2.0
    b = (nz - 2) / 2.0
    y = js - (ny - 1) / 2.0
    z = k - (nz - 1) / 2.0
    u = np.zeros_like(y, dtype=np.float64)
    for n in range(1, 2 * terms, 2):
        q = n * np.pi / (2.0 * a)
        u += ((-1) ** ((n - 1) // 2) / n ** 3) * (1.0 - np.cosh(q * z) / np.cosh(q * b)) * np.cos(q * y)
    return u


def run(n_steps: int = 3000, threshold: float = 5.0) -> float:
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
        g.T_max, g.T_prev, g.dT_dt,
        g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
        g.Fx_snap, g.Fy_snap, g.Fz_snap,
        g.Fx, g.Fy, g.Fz, twin.T_amb,
        g.porosity_active, g.tracer_head, g.max_tracers,
    )
    kernels.stream(g.f_a, g.f_b, g.flags, g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz, 1)

    js = np.arange(1, ny - 1, dtype=np.float64)
    u_anal = _duct_profile(js, nz // 2, ny, nz)

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
            g.nx, g.ny, g.nz, 1,
        )

    ux_np = g.ux.to_numpy()
    flags_np = g.flags.to_numpy()
    mid_i = nx // 2
    mid_k = nz // 2

    u_meas = np.array([
        float(ux_np[mid_i, j, mid_k])
        for j in range(1, ny - 1)
        if flags_np[mid_i, j, mid_k] == g.FLAG_FLUID
    ])

    u_meas_n = u_meas / (np.max(u_meas) + 1e-12)
    u_anal_n = u_anal / (np.max(u_anal) + 1e-12)
    l2 = float(np.sqrt(np.mean((u_meas_n - u_anal_n) ** 2)) * 100.0)

    print(f"[lbm_poiseuille] profile L2 = {l2:.2f}%  (threshold {threshold}%)")
    if l2 >= threshold:
        raise AssertionError(f"Poiseuille profile L2 {l2:.2f}% >= {threshold}%")
    return l2


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
