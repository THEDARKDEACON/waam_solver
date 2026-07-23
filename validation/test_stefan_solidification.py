"""
test_stefan_solidification.py — 1D solidification front vs Stefan sqrt(t) scaling.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.materials import load_material
from waam_twin import kernels


def solve_stefan_lambda(ste: float, tol: float = 1e-8) -> float:
    """Solve sqrt(pi)*lam*exp(lam^2)*erf(lam) = Ste for lam > 0."""

    def residual(lam: float) -> float:
        return math.sqrt(math.pi) * lam * math.exp(lam * lam) * math.erf(lam) - ste

    lo, hi = 1e-6, 5.0
    while residual(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Failed to bracket Stefan lambda root")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def interface_height_m(
    fl_np: np.ndarray,
    T_np: np.ndarray,
    flags_np: np.ndarray,
    nz_solid: int,
    dx: float,
    T_solidus: float,
    T_liquidus: float,
    flag_fluid: int,
) -> float:
    """Distance of T=(T_s+T_l)/2 isotherm above the substrate–liquid boundary."""
    mid_i = fl_np.shape[0] // 2
    mid_j = fl_np.shape[1] // 2
    col_fl = fl_np[mid_i, mid_j, :]
    col_T = T_np[mid_i, mid_j, :]
    col_flags = flags_np[mid_i, mid_j, :]
    T_mid = 0.5 * (T_solidus + T_liquidus)

    for k in range(nz_solid, fl_np.shape[2] - 1):
        if col_flags[k] != flag_fluid:
            continue
        t_lo = float(col_T[k])
        t_hi = float(col_T[k + 1])
        if t_lo < T_mid <= t_hi:
            frac = (T_mid - t_lo) / max(t_hi - t_lo, 1e-9)
            return (k - nz_solid + frac) * dx
        if t_lo >= T_mid > t_hi:
            frac = (t_lo - T_mid) / max(t_lo - t_hi, 1e-9)
            return (k - nz_solid + 1.0 - frac) * dx

    for k in range(nz_solid, fl_np.shape[2] - 1):
        if col_flags[k] != flag_fluid:
            continue
        f_lo = float(col_fl[k])
        f_hi = float(col_fl[k + 1])
        if f_lo < 0.5 <= f_hi:
            frac = (0.5 - f_lo) / max(f_hi - f_lo, 1e-12)
            return (k - nz_solid + frac) * dx

    if col_fl[nz_solid] >= 0.5:
        return 0.0
    return (fl_np.shape[2] - nz_solid) * dx


def run(n_steps: int = 800, threshold: float = 8.0) -> float:
    # Gate tightened 15% → 10% after the latent-heat advection fix (measured 6.0%).
    init_taichi(backend="cpu")
    mat = load_material("materials/placeholders/ER70S-6.yaml")

    nx, ny, nz = 4, 4, 80
    nz_solid = 8
    dx = 1.0e-4
    T_amb = 300.0
    T_init = mat.T_liquidus + 50.0

    twin = WAAMTwin(
        material=mat,
        nx=nx, ny=ny, nz=nz,
        dx=dx,
        C_darcy=0.0,
        max_tracers=10,
    )
    g = twin.grid

    kernels.init_stefan_liquid_column(
        g.T, g.H, g.f_l, g.phi, g.flags,
        nz_solid, T_amb, T_init,
        twin.cp_rho, twin.L_rho,
        mat.T_solidus, mat.T_liquidus,
        g.FLAG_FLUID, g.FLAG_SOLID,
    )

    ste = mat.cp * (T_init - T_amb) / mat.L_fusion
    lam = solve_stefan_lambda(ste)
    slope_anal = 2.0 * lam * math.sqrt(mat.alpha)

    sample_steps = [200, 300, 400, 500, 600, 700]
    sqrt_t: list[float] = []
    s_sim_hist: list[float] = []

    for step in range(1, n_steps + 1):
        kernels.clamp_substrate_enthalpy(
            g.H, g.T, g.f_l, g.flags,
            nz_solid, T_amb, twin.cp_rho, g.FLAG_SOLID,
        )
        kernels.advect_diffuse_temperature(
            g.H, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz,
        )
        kernels.update_phase(
            g.H, g.T, g.f_l,
            twin.cp_rho, twin.L_rho,
            mat.T_solidus, mat.T_liquidus,
        )

        if step in sample_steps:
            t_phys = step * g.dt
            s_sim = interface_height_m(
                g.f_l.to_numpy(),
                g.T.to_numpy(),
                g.flags.to_numpy(),
                nz_solid, dx,
                mat.T_solidus, mat.T_liquidus,
                g.FLAG_FLUID,
            )
            sqrt_t.append(math.sqrt(t_phys))
            s_sim_hist.append(s_sim)

    if len(sqrt_t) < 3:
        raise AssertionError("Insufficient Stefan samples")

    if not all(s_sim_hist[i] <= s_sim_hist[i + 1] for i in range(len(s_sim_hist) - 1)):
        raise AssertionError("Solidification front must advance monotonically")

    slope_sim = float(np.polyfit(sqrt_t, s_sim_hist, 1)[0])
    slope_err = abs(slope_sim - slope_anal) / slope_anal * 100.0

    print(
        f"[stefan_solidification] slope error = {slope_err:.2f}%  "
        f"(sim={slope_sim:.4e} anal={slope_anal:.4e}  threshold {threshold}%)"
    )
    if slope_err >= threshold:
        raise AssertionError(f"Stefan slope error {slope_err:.2f}% >= {threshold}%")
    return slope_err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
