"""
test_thermal_diffusion.py — kernel-only Gaussian pulse diffusion (no full step).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.materials import load_material
from waam_twin import kernels


def gaussian_diffusion_analytical(R, T_bg, A, sigma0, alpha, t):
    sigma_sq = sigma0 ** 2 + 2.0 * alpha * t
    amplitude_decay = (sigma0 ** 2 / sigma_sq) ** 1.5
    return T_bg + A * amplitude_decay * np.exp(-R ** 2 / (2.0 * sigma_sq))


def run(n_steps: int = 400, nx: int = 48, threshold: float = 12.0) -> float:
    init_taichi(backend="cpu")
    mat = load_material("materials/placeholders/ER70S-6.yaml")
    T_bg = 300.0
    dx = 3e-4
    sigma0_m = 1.2e-3
    A_K = 500.0

    twin = WAAMTwin(
        material=mat,
        nx=nx, ny=nx, nz=nx,
        dx=dx,
        use_srt=True,
        C_darcy=0.0,
        max_tracers=100,
    )
    twin.reset(T_ambient=T_bg, test_fluid_domain=True)
    g = twin.grid

    cx = (nx // 2) * dx
    kernels.prescribe_gaussian_pulse(
        g.T, g.H, g.f_l, g.flags,
        T_bg, A_K, sigma0_m, cx, cx, cx, dx,
        twin.cp_rho, g.FLAG_GAS,
    )

    for _ in range(n_steps):
        kernels.advect_diffuse_temperature(
            g.H, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz,
        )
        kernels.sync_T_from_H(g.H, g.T, twin.cp_rho)

    t_sim = n_steps * g.dt
    T_sim = g.T.to_numpy()

    x_c = (np.arange(nx) - nx // 2) * dx
    XX, YY, ZZ = np.meshgrid(x_c, x_c, x_c, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
    T_anal = gaussian_diffusion_analytical(R, T_bg, A_K, sigma0_m, mat.alpha, t_sim)

    interior = np.s_[4:-4, 4:-4, 4:-4]
    T_sim_i = T_sim[interior]
    T_anal_i = T_anal[interior]
    l2 = float(
        np.sqrt(np.sum((T_sim_i - T_anal_i) ** 2) / np.sum((T_anal_i - T_bg) ** 2)) * 100.0
    )

    print(f"[thermal_diffusion] L2 error = {l2:.2f}%  (threshold {threshold}%)")
    if l2 >= threshold:
        raise AssertionError(f"L2 {l2:.2f}% >= {threshold}%")
    return l2


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
