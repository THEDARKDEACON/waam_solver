"""
units.py — Single SI ↔ lattice conversion module (Phase 2).

All physical inputs in job/material YAML are SI. Lattice-unit knobs must not
appear in jobs; convert here only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CS2 = 1.0 / 3.0  # D3Q19 lattice speed of sound squared


@dataclass(frozen=True)
class LatticeScale:
    """Physical ↔ lattice scale factors for one mesh / timestep."""

    dx: float  # m
    dt: float  # s
    rho: float  # kg/m³ reference

    @property
    def force_scale(self) -> float:
        """Physical force density [N/m³] → lattice force density (ρ_lu≈1)."""
        return (self.dt ** 2) / (self.rho * self.dx ** 2)

    @property
    def surface_force_scale(self) -> float:
        """Surface tension / Marangoni prefactor (includes 1/dx for lattice grads)."""
        return self.force_scale / self.dx

    def g_lu(self, g_phys: float = 9.81) -> float:
        return g_phys * (self.dt ** 2) / self.dx

    def nu_lu(self, nu_phys: float) -> float:
        return nu_phys * self.dt / (self.dx ** 2)

    def tau_from_nu(self, nu_phys: float) -> float:
        return 3.0 * self.nu_lu(nu_phys) + 0.5

    def alpha_lu(self, alpha_phys: float) -> float:
        return alpha_phys * self.dt / (self.dx ** 2)

    def u_lu(self, u_phys: float) -> float:
        return u_phys * self.dt / self.dx

    def u_phys(self, u_lu: float) -> float:
        return u_lu * self.dx / self.dt

    def pressure_to_Fz_lu(self, pressure_pa: float) -> float:
        """Surface pressure [Pa] → lattice body-force density on a 1-cell band."""
        return (pressure_pa / (self.rho * self.dx)) * (self.dt ** 2) / self.dx

    def C_darcy_lu(self, C_phys: float) -> float:
        """
        Physical Carman–Kozeny [kg/(m³·s)] → lattice coefficient.

        In Guo form: K_lu = C_phys * dt / rho  (dimensionless per step).
        """
        return C_phys * self.dt / self.rho

    def C_phys_from_lu(self, C_lu: float) -> float:
        return C_lu * self.rho / self.dt


def choose_dt(
    dx: float,
    *,
    nu_phys: float,
    u_design_m_s: float = 1.0,
    ma_max: float = 0.08,
    tau_min: float = 0.505,
    mode: str = "acoustic",
) -> float:
    """
    Choose physical dt [s].

    mode:
      acoustic  — Ma-limited (dt ∝ dx), production default
      diffusive — dt ∝ dx² for mesh-convergence studies (M13)
    """
    u_des = max(float(u_design_m_s), 1e-9)
    u_cap = max(float(ma_max), 1e-9)
    dt_ma = u_cap * dx / u_des

    # τ = 3 ν dt/dx² + 1/2 ≥ tau_min → dt ≥ (tau_min-0.5) dx² / (3ν)
    nu = max(float(nu_phys), 1e-20)
    dt_tau = (tau_min - 0.5) * (dx ** 2) / (3.0 * nu)

    if mode == "diffusive":
        # Prefer τ target; still respect Ma
        dt = max(dt_tau, 1e-12)
        if dt > dt_ma:
            raise ValueError(
                f"diffusive dt={dt:.3e}s violates Ma cap dt≤{dt_ma:.3e}s; "
                f"increase dx or viscosity, or raise ma_max"
            )
        return dt

    # acoustic: start from Ma, raise to meet τ if possible
    dt = dt_ma
    if dt < dt_tau:
        if dt_tau > dt_ma + 1e-30:
            raise ValueError(
                f"cannot satisfy τ≥{tau_min} and Ma≤{ma_max} together "
                f"(need dt≥{dt_tau:.3e}s for τ, dt≤{dt_ma:.3e}s for Ma)"
            )
        dt = dt_tau
    return dt


def pressure_force_normalized(
    pressure_pa: float,
    grad_phi_mag: float,
    scale: LatticeScale,
) -> float:
    """
    Mesh-independent surface load: F = P · |∇φ| · scale.

    Integrates to ~P over the interface band regardless of smearing width.
    """
    return pressure_pa * grad_phi_mag * scale.force_scale / scale.dx


LATTICE_JOB_KEYS = frozenset(
    {
        "force_limit_lu",
        "u_mach_limit_lu",
        "C_darcy",
        "C_darcy_ref_dt_s",
        "dt_scale",
        "gamma_lu",
        "dgamma_dT_lu",
        "g_lu",
        "omega",
        "omega_bulk",
    }
)


def assert_no_lattice_job_keys(sim: dict) -> None:
    """Raise if a job simulation block still carries lattice-unit knobs."""
    bad = sorted(k for k in sim if k in LATTICE_JOB_KEYS)
    if bad:
        raise ValueError(
            f"Job simulation contains lattice-unit knobs {bad}. "
            f"Express limits in SI (force_limit_m_s2, u_design_m_s, "
            f"C_darcy_phys) and convert via waam_twin.units."
        )
