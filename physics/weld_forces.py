"""Orchestration for advanced weld body forces (recoil, gas shear, Lorentz, droplet)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .deposition_balance import droplet_mass_kg

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid

MU0 = 4.0e-7 * math.pi


def _kwf():
    from .. import kernels
    return kernels


def droplet_impact_velocity_m_s(twin: "WAAMTwin") -> float:
    """Impact speed from wire feed + gravitational detachment (deterministic)."""
    g = 9.81
    drop_len = 3e-3
    if twin.droplet_freq > 0 and twin.wire_feed_m_s > 0:
        drop_len = twin.wire_feed_m_s / twin.droplet_freq
    v_feed = max(twin.wire_feed_m_s, 0.0)
    v_grav = math.sqrt(max(0.0, 2.0 * g * drop_len))
    return min(3.0, max(v_feed, v_grav))


def droplet_velocity_lu(twin: "WAAMTwin") -> tuple[float, float, float]:
    """Droplet velocity in lattice units (downward + travel direction)."""
    grid = twin.grid
    v = droplet_impact_velocity_m_s(twin)
    vz_lu = -v * grid.dt / grid.dx
    vx_lu = twin.travel_speed_m_s * grid.dt / grid.dx
    return vx_lu, 0.0, vz_lu


def droplet_impact_pressure_pa(twin: "WAAMTwin") -> float:
    """Kinetic pressure scale ½ρv² from droplet mass and contact area."""
    m = droplet_mass_kg(twin)
    if m <= 0:
        return 0.0
    v = droplet_impact_velocity_m_s(twin)
    r = max(twin.grid.dx * 1.5, twin.wire_diameter_m * 0.5)
    area = math.pi * r * r
    ke = 0.5 * m * v * v
    return ke / max(area, 1e-12)


def gas_shear_tau_pa(twin: "WAAMTwin") -> float:
    """Dynamic shear ~ ρ_gas v² (capped)."""
    rho_gas = 1.6  # argon kg/m³
    v = twin.gas_jet_velocity_m_s
    tau = 0.5 * rho_gas * v * v * twin.gas_shear_coeff
    return min(tau, 5000.0)


def apply_recoil(twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
    if not twin.enable_recoil:
        return
    if twin.use_recoil_clausius_clapeyron:
        _kwf().apply_vapor_recoil_clausius_clapeyron(
            g.Fz, g.T, g.phi, g.flags,
            arc_i, arc_j, arc_k, twin.sigma_cells,
            twin.P_vapor_ref_Pa, twin.T_boiling_K, twin.L_vapor_J_kg, twin.R_spec_vapor_J_kgK,
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_GAS, g.FLAG_SOLID,
        )
    else:
        from .forces import apply_vapor_recoil
        apply_vapor_recoil(
            g.Fz, g.T, g.phi, g.flags,
            arc_i, arc_j, arc_k, twin.sigma_cells,
            twin.recoil_pressure_pa, twin.mat.T_liquidus,
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_GAS, g.FLAG_SOLID,
        )


def apply_gas_shear(twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
    if not twin.enable_gas_shear:
        return
    tau = gas_shear_tau_pa(twin)
    kwf = _kwf()
    kwf.apply_gas_shear_stress(
        g.Fx, g.Fy, g.phi, g.flags,
        arc_i, arc_j, arc_k, twin.sigma_cells,
        tau, g.dt, g.dx, twin.mat.rho,
        g.FLAG_GAS, g.FLAG_SOLID,
    )


def apply_droplet_impact(
    twin: "WAAMTwin",
    g: "WAAMGrid",
    arc_i: float,
    arc_j: float,
    arc_k: float,
    drop_r: float,
) -> None:
    if not twin.enable_deposition_momentum:
        return
    vx, vy, vz = droplet_velocity_lu(twin)
    kwf = _kwf()
    kwf.feed_wire_momentum_impact(
        g.ux, g.uy, g.uz, g.flags, g.f_l,
        arc_i, arc_j, arc_k, drop_r,
        vx, vy, vz,
        g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    if twin.enable_droplet_impact_pressure:
        kwf.apply_droplet_impact_pressure(
            g.Fz, g.flags, g.phi, g.f_l,
            arc_i, arc_j, arc_k, drop_r,
            droplet_impact_pressure_pa(twin),
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_GAS, g.FLAG_SOLID,
        )


def solve_lorentz(twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
    if not twin.enable_lorentz:
        return
    kwf = _kwf()
    kwf.elec_build_sigma(
        g.sigma_elec, g.f_l, g.flags,
        twin.sigma_liquid_Sm, twin.sigma_solid_Sm,
        g.FLAG_GAS,
    )
    kwf.elec_clear_source(g.elec_source)
    kwf.elec_inject_arc_source(
        g.elec_source, arc_i, arc_j, arc_k,
        twin.sigma_cells, twin.welding_current_A, g.dx, g.nz,
    )
    kwf.elec_normalize_source(g.elec_source, twin.welding_current_A, g.dx)
    kwf.elec_init_ground(g.phi_elec, g.flags, twin.nz_solid, g.FLAG_SOLID)
    for _ in range(twin.lorentz_jacobi_iters):
        kwf.elec_jacobi_step(
            g.phi_elec, g.phi_elec_tmp, g.sigma_elec, g.elec_source, g.flags,
            g.dx, twin.lorentz_jacobi_omega,
            g.FLAG_GAS, g.FLAG_SOLID, twin.nz_solid, g.nx, g.ny, g.nz,
        )
        g.phi_elec.copy_from(g.phi_elec_tmp)
    kwf.elec_compute_J(
        g.Jx, g.Jy, g.Jz, g.phi_elec, g.sigma_elec, g.flags, g.dx,
        g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    kwf.elec_compute_B_from_J(
        g.Bx, g.By, g.Bz, g.Jx, g.Jy, g.Jz, g.dx, MU0, g.nx, g.ny, g.nz,
    )
    kwf.apply_lorentz_JxB(
        g.Fx, g.Fy, g.Fz,
        g.Jx, g.Jy, g.Jz, g.Bx, g.By, g.Bz,
        g.f_l, g.flags, g.dt, g.dx, twin.mat.rho, g.FLAG_GAS,
    )
