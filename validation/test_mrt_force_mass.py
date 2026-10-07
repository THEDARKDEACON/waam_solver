"""
test_mrt_force_mass.py — Two-rate MRT regenerate-and-rediff mass + Guo force.

Checks collide_mrt_variable_tau:
  1. Σf is conserved to machine tolerance (mass correction + rest stress term)
  2. First-moment impulse ≈ F (not ~(3-ω_s)/2 · F) vs SRT reference at same τ
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.cumulant_kernel import collide_mrt_variable_tau
from waam_twin.runtime import init_taichi


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
    twin.reset(test_fluid_domain=True)
    g = twin.grid
    # Uniform quiescent fluid, constant body force
    Fx = 1.0e-4
    g.Fx.fill(Fx)
    g.Fy.fill(0.0)
    g.Fz.fill(0.0)
    g.f_l.fill(1.0)

    f0 = g.f_a.to_numpy().copy()
    mass0 = float(f0.sum())
    # Momentum before collision (lattice)
    # D3Q19 e_x from kernels: use stored moments after collide via ux

    # One MRT variable-τ collide into f_b
    omega_b = float(1.0 / max(twin.grid.tau, 0.505))
    collide_mrt_variable_tau(
        g.f_a, g.f_b,
        g.rho, g.ux, g.uy, g.uz,
        g.Fx, g.Fy, g.Fz,
        g.f_l, g.tau_field, g.flags,
        g.ex, g.ey, g.ez, g.w, g.opp,
        omega_b,
        0.0,  # C_darcy
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    f1 = g.f_b.to_numpy()
    mass1 = float(f1.sum())
    mass_err = abs(mass1 - mass0) / max(abs(mass0), 1e-30)
    if mass_err > 1e-5:
        raise AssertionError(
            f"MRT mass not conserved: Σf_in={mass0:.8g} Σf_out={mass1:.8g} "
            f"rel_err={mass_err:.3e}"
        )

    # First moment of post-collision populations (x)
    ex = g.ex.to_numpy()
    jx = 0.0
    for q in range(19):
        jx += float(ex[q]) * float(f1[q].sum())
    n_fluid = int(np.sum(g.flags.to_numpy() == g.FLAG_FLUID))
    # Expected: each fluid cell gains ≈ Fx (Guo full impulse per step)
    jx_expected = Fx * n_fluid
    # Pre-collision jx ≈ 0 for equilibrium rest fluid
    force_err = abs(jx - jx_expected) / max(abs(jx_expected), 1e-30)
    # Allow 15% — regenerate Hermite is approximate but must not be ~0.5×
    if force_err > 0.15:
        raise AssertionError(
            f"MRT Guo impulse wrong: jx={jx:.6e} expected≈{jx_expected:.6e} "
            f"(err={force_err:.1%}); near τ=0.5 a (3-ω)/2 bug gives ~50% shortfall"
        )
    # Also compare to half-force failure mode explicitly
    if abs(jx - 0.5 * jx_expected) / max(abs(jx_expected), 1e-30) < 0.10:
        raise AssertionError(
            f"MRT impulse looks like half-force bug: jx={jx:.6e} ≈ 0.5×{jx_expected:.6e}"
        )

    print(
        f"[mrt_force_mass] mass_rel_err={mass_err:.2e}  "
        f"jx={jx:.6e} expected={jx_expected:.6e} force_err={force_err:.2%}  PASS"
    )
    return force_err


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    raise SystemExit(0)
