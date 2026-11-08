"""
test_surface_load_integral.py — arc surface loads integrate to ∫P dA.

Arc pressure, vapor recoil (Clausius–Clapeyron) and gas shear are surface
loads from above. On a flat pool surface their total force must equal the
area integral of the applied pressure (shear stress for gas shear), for a
sharp interface (top cell full) and a resolved one (top cell 40 % full),
and on two meshes.

Regression: arc pressure gave the full P to every cell with |Δφ_z| > 0.05
(≈2× ∫P dA on a resolved interface); CC recoil used central |∇φ| that skips
the gas side (50 % of ∫P dA on a sharp interface, 80 % at fill 0.4).
"""

from __future__ import annotations

import math
import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.kernels import weld as kw
from waam_twin.runtime import init_taichi

P_PA = 1000.0
SIGMA_MM = 1.5
TOL = 0.02


def _setup(dx: float, top_fill: float) -> tuple[WAAMTwin, int]:
    n = int(round(12e-3 / dx))
    twin = WAAMTwin(nx=n, ny=n, nz=12, dx=dx, enable_vof=True, max_tracers=10)
    twin.reset(test_fluid_domain=True)
    g = twin.grid
    k_top = 6
    phi = np.zeros((g.nx, g.ny, g.nz), dtype=np.float32)
    phi[:, :, :k_top] = 1.0
    phi[:, :, k_top] = top_fill
    flags = np.where(phi > 0.0, g.FLAG_FLUID, g.FLAG_GAS).astype(np.int32)
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)
    g.T.fill(3500.0)  # above boiling: CC recoil pressure is uniform in T
    g.alloy_frac.fill(0.0)
    return twin, k_top


def _total_N(twin: WAAMTwin, comp: str) -> float:
    """Σ F_lu converted to newtons: F_phys = F_lu·dx/dt², force = ρ dx³ F_phys."""
    g = twin.grid
    rho = float(twin.mat.rho)
    f = getattr(g, comp).to_numpy().astype(np.float64)
    return float(f.sum()) * rho * g.dx ** 4 / g.dt ** 2


def _clear(g) -> None:
    g.Fx.fill(0.0)
    g.Fy.fill(0.0)
    g.Fz.fill(0.0)


def run() -> float:
    init_taichi(backend="cpu")
    worst = 0.0
    for dx in (4e-4, 2.5e-4):
        for fill in (1.0, 0.4):
            twin, k_top = _setup(dx, fill)
            g = twin.grid
            rho = float(twin.mat.rho)
            sig = SIGMA_MM * 1e-3 / dx
            ci = cj = g.nx / 2.0
            I, J = np.meshgrid(np.arange(g.nx), np.arange(g.ny), indexing="ij")
            gauss = np.exp(-((I - ci) ** 2 + (J - cj) ** 2) / (2.0 * sig * sig))
            expect = P_PA * float(gauss.sum()) * dx * dx  # ∫P dA on the grid

            _clear(g)
            kw.apply_arc_pressure(
                g.Fz, g.flags, g.phi, ci, cj, float(k_top), sig, P_PA,
                g.dt, g.dx, g.alloy_frac, rho, rho, g.FLAG_SOLID, g.FLAG_GAS,
            )
            F_arc = -_total_N(twin, "Fz")

            # CC recoil with T above boiling → P_vap = C_acc · P_ref · e^{L/R(1/Tb − 1/T)}.
            _clear(g)
            T_b, L, R, C = 3000.0, 6.0e6, 149.0, 0.5
            P_ref = P_PA / (C * math.exp(min(12.0, (L / R) * (1.0 / T_b - 1.0 / 3500.0))))
            kw.apply_vapor_recoil_clausius_clapeyron(
                g.Fz, g.T, g.phi, g.flags, ci, cj, float(k_top), sig,
                P_ref, T_b, 2800.0, L, R, C,
                g.dt, g.dx, g.alloy_frac, rho, rho, g.FLAG_SOLID, g.FLAG_GAS,
            )
            F_rec = -_total_N(twin, "Fz")

            _clear(g)
            kw.apply_gas_shear_stress(
                g.Fx, g.Fy, g.phi, g.flags, ci, cj, float(k_top), sig, P_PA,
                g.dt, g.dx, g.alloy_frac, rho, rho, g.FLAG_SOLID, g.FLAG_GAS,
            )
            fx = g.Fx.to_numpy().astype(np.float64)
            fy = g.Fy.to_numpy().astype(np.float64)
            F_shear = float(np.sqrt(fx ** 2 + fy ** 2).sum()) * rho * dx ** 4 / g.dt ** 2

            for name, F in (("arc", F_arc), ("recoil", F_rec), ("shear", F_shear)):
                err = abs(F / expect - 1.0)
                worst = max(worst, err)
                print(
                    f"[surface_load] dx={dx * 1e3:.2f}mm fill={fill:.1f} {name:<6} "
                    f"ΣF={F:.4e} N  ∫P dA={expect:.4e} N  ratio={F / expect:.4f}"
                )
                if err > TOL:
                    raise AssertionError(
                        f"{name} load at dx={dx} fill={fill}: ΣF/∫P dA = {F / expect:.4f} "
                        f"(tol ±{TOL:.0%})"
                    )
    print(f"[surface_load] worst |ratio − 1| = {worst:.2e}  PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
