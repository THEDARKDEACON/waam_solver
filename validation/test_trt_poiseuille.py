"""
test_trt_poiseuille.py — force-driven duct flow on the production TRT path.

collide_mrt with omega_b = 0 (magic Λ = 1/4) is started from the analytic
duct solution and must hold it under the body force. The production shop
mesh runs at τ_s ≈ 0.5007, where a viscous decay from rest is too slow to
reach steady state in a test, so holding the steady state is what is checked.

Regression: with the whole Guo source scaled by (1 − ω_e/2), the force
reaches only ~4 % (τ_s = 0.51) / ~0.3 % (τ_s = 0.5007) of its value and the
profile decays viscously (≈40 % and ≈4 % over these runs).

Cases are kept at low Reynolds number (Re = u_max·H/ν ≲ 160). At τ_s = 0.5007
and Re ~ 10³ the 3-D duct develops transverse motion that dissipates the mean
profile, which is a flow-regime effect, not a forcing error.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin import kernels
from waam_twin.collision import collide_mrt
from waam_twin.lattice import EX, W
from waam_twin.runtime import init_taichi

# (τ_s, u_max_lu, steps, omega_b): omega_b 0 → magic Λ = 1/4; 1.0 → τ_odd = 1,
# the twin default since Nov 2026.
CASES = (
    (0.51, 0.01, 3000, 0.0),
    (0.5007, 0.002, 4000, 0.0),
    (0.51, 0.01, 3000, 1.0),
    (0.5007, 0.002, 4000, 1.0),
)
TOL = 0.015  # max |u − u_anal| / u_max (magic Λ = 1/4: bounce-back wall ~halfway)
# τ_odd = 1 gives Λ = (τ_s − ½)/2 ≈ 4e-4 at τ_s 0.5007, so the bounce-back wall
# sits slightly off halfway and a 16-cell duct runs ~1.4 % fast. Accepted for
# stability (magic Λ blows up under mushy-zone drag); still tracked here.
TOL_TAU_ODD_1 = 0.025


def duct_velocity(ny: int, nz: int, F: float, nu: float, terms: int = 41) -> np.ndarray:
    """Analytic duct velocity with half-way bounce-back walls (cells 0 and n−1 solid)."""
    a = (ny - 2) / 2.0
    b = (nz - 2) / 2.0
    y = np.arange(ny)[:, None] - (ny - 1) / 2.0
    z = np.arange(nz)[None, :] - (nz - 1) / 2.0
    s = np.zeros((ny, nz))
    for n in range(1, 2 * terms, 2):
        q = n * np.pi / (2.0 * a)
        s += ((-1) ** ((n - 1) // 2) / n ** 3) * (
            1.0 - np.cosh(q * z) / np.cosh(q * b)
        ) * np.cos(q * y)
    u = 16.0 * a * a / np.pi ** 3 * s * F / nu
    u[0, :] = u[-1, :] = 0.0
    u[:, 0] = u[:, -1] = 0.0
    return u


def _run_case(twin: WAAMTwin, tau: float, u_max: float, steps: int, omega_b: float = 0.0) -> float:
    g = twin.grid
    nx, ny, nz = g.nx, g.ny, g.nz
    nu = (tau - 0.5) / 3.0
    shape = duct_velocity(ny, nz, 1.0, 1.0)
    F = float(u_max * nu / shape.max())
    U = duct_velocity(ny, nz, F, nu)

    kernels.init_poiseuille_channel(
        g.f_a, g.rho, g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
        1.0, ny, nz, 0.0, nu, g.FLAG_FLUID, g.FLAG_SOLID,
    )
    f0 = np.zeros((19, nx, ny, nz), dtype=np.float32)
    for q in range(19):
        eu = EX[q] * U
        f0[q] = (W[q] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * U * U))[None]
    g.f_a.from_numpy(f0)

    for _ in range(steps):
        kernels.set_uniform_Fx(g.Fx, g.Fy, g.Fz, F, g.flags, g.FLAG_FLUID)
        collide_mrt(
            g.f_a, g.f_b, g.rho, g.ux, g.uy, g.uz, g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags, g.ex, g.ey, g.ez, g.w, g.opp,
            1.0 / tau, omega_b,  # omega_s, omega_b (0 → magic Λ = 1/4)
            0.0,             # C_darcy
            g.FLAG_SOLID, g.FLAG_GAS, nx, ny, nz,
        )
        kernels.stream(g.f_b, g.f_a, g.flags, g.FLAG_SOLID, g.FLAG_GAS, nx, ny, nz, 1)

    ux = g.ux.to_numpy()[nx // 2]
    fluid = g.flags.to_numpy()[nx // 2] == g.FLAG_FLUID
    err = float(np.abs(ux[fluid] - U[fluid]).max() / U.max())
    ratio = float(ux[fluid].max() / U.max())
    print(
        f"[trt_poiseuille] τ_s={tau:<6g} {'magic' if omega_b <= 0 else f'ω_o={omega_b:g}':>7s} Re={u_max * (ny - 2) / nu:6.0f} steps={steps}  "
        f"u_max/anal={ratio:.4f}  max err={err * 100:.2f}%"
    )
    return err


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=4, ny=18, nz=18, dx=3e-4,
        use_srt=False, C_darcy=0.0, max_tracers=10,
    )
    worst = 0.0
    for tau, u_max, steps, omega_b in CASES:
        err = _run_case(twin, tau, u_max, steps, omega_b)
        worst = max(worst, err)
        tol = TOL if omega_b <= 0.0 else TOL_TAU_ODD_1
        if err > tol:
            raise AssertionError(
                f"TRT duct at τ_s={tau} drifted {err * 100:.2f}% from the analytic "
                f"profile (tol {tol * 100:.1f}%): body force not fully applied"
            )
    print(f"[trt_poiseuille] worst error {worst * 100:.2f}%  PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
