"""
test_mrt_force_mass.py — TRT (two-rate) mass + Guo force.

Checks collide_mrt_variable_tau on a quiescent equilibrium fluid:
  1. Σf conserved to machine tolerance (TRT conserves mass by construction)
  2. First-moment impulse of one collision equals F, on the magic Λ = 1/4
     branch the production path uses (omega_b <= 0), at τ_s down to the
     production 0.5007, and on the explicit-ω_o branch.

Regression: scaling the whole Guo source by (1 − ω_e/2) gives an impulse of
F·(1 + (ω_o − ω_e)/2) — 0.3 % of F at τ_s = 0.5007. The earlier version of
this test passed omega_b = 1/max(τ, 0.505), which missed the magic branch.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.cumulant_kernel import collide_mrt_variable_tau
from waam_twin.runtime import init_taichi

# (τ_s, omega_b): omega_b <= 0 → magic Λ = 1/4; > 0 → explicit odd rate.
CASES = (
    (0.5007, 0.0),
    (0.51, 0.0),
    (0.6, 0.0),
    (1.0, 0.0),
    (0.5007, 1.0),
)
FORCE_TOL = 0.01


def _impulse(twin: WAAMTwin, tau_s: float, omega_b: float, Fx: float) -> tuple[float, float]:
    """Return (mass_rel_err, jx / (Fx · n_fluid)) for one collision from rest."""
    g = twin.grid
    twin.reset(test_fluid_domain=True)
    g.Fx.fill(Fx)
    g.Fy.fill(0.0)
    g.Fz.fill(0.0)
    g.f_l.fill(1.0)
    g.tau_field.fill(tau_s)

    f0 = g.f_a.to_numpy().astype(np.float64)
    mass0 = float(f0.sum())
    collide_mrt_variable_tau(
        g.f_a, g.f_b,
        g.rho, g.ux, g.uy, g.uz,
        g.Fx, g.Fy, g.Fz,
        g.f_l, g.tau_field, g.flags,
        g.ex, g.ey, g.ez, g.w, g.opp,
        float(omega_b),
        0.0,  # C_darcy
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
        0.0,  # cs_smag
    )
    f1 = g.f_b.to_numpy().astype(np.float64)
    mass_err = abs(float(f1.sum()) - mass0) / max(abs(mass0), 1e-30)

    ex = g.ex.to_numpy()
    jx0 = sum(float(ex[q]) * float(f0[q].sum()) for q in range(19))
    jx1 = sum(float(ex[q]) * float(f1[q].sum()) for q in range(19))
    n_fluid = int(np.sum(g.flags.to_numpy() == g.FLAG_FLUID))
    return mass_err, (jx1 - jx0) / (Fx * n_fluid)


def run() -> float:
    init_taichi(backend="cpu")
    n = 12
    twin = WAAMTwin(
        nx=n, ny=n, nz=n, dx=5e-4,
        use_srt=False, use_variable_tau=True,
        C_darcy=0.0, max_tracers=10,
        force_limit_lu=10.0, u_mach_limit_lu=0.5,
        enable_vof=False,
    )
    Fx = 1.0e-4
    worst = 0.0
    for tau_s, omega_b in CASES:
        mass_err, ratio = _impulse(twin, tau_s, omega_b, Fx)
        branch = "magic" if omega_b <= 0.0 else f"ω_o={omega_b:g}"
        print(
            f"[mrt_force_mass] τ_s={tau_s:<6g} {branch:<9} "
            f"mass_rel_err={mass_err:.2e}  impulse/F={ratio:.5f}"
        )
        if mass_err > 1e-5:
            raise AssertionError(
                f"TRT mass not conserved at τ_s={tau_s}: rel_err={mass_err:.3e}"
            )
        err = abs(ratio - 1.0)
        worst = max(worst, err)
        if err > FORCE_TOL:
            raise AssertionError(
                f"TRT Guo impulse wrong at τ_s={tau_s} ({branch}): "
                f"impulse/F={ratio:.5f}, expected 1 ± {FORCE_TOL}. A single "
                f"(1 − ω_e/2) prefactor on the odd source gives "
                f"1 + (ω_o − ω_e)/2."
            )
    print(f"[mrt_force_mass] worst |impulse/F − 1| = {worst:.2e}  PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    raise SystemExit(0)
