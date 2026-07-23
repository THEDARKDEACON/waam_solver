"""Orchestration for advanced weld body forces (recoil, gas shear, Lorentz, droplet)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .deposition_balance import droplet_mass_kg, infer_transfer_mode

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid

MU0 = 4.0e-7 * math.pi


def _kwf():
    from .. import kernels
    return kernels


def lin_eagar_peak_pa(current_A: float, sigma_p_m: float) -> float:
    """
    Lin & Eagar (1986) peak arc pressure [Pa]:

        p0 = μ0 · I² / (4 π² σ_p²)

    σ_p is the Gaussian pressure radius [m].
    """
    sig = max(float(sigma_p_m), 1.0e-6)
    i = max(float(current_A), 0.0)
    return MU0 * i * i / (4.0 * math.pi * math.pi * sig * sig)


def arc_pressure_sigma_m(twin: "WAAMTwin") -> float:
    """Pressure Gaussian radius [m]; defaults to arc heat σ."""
    sig = getattr(twin, "pressure_sigma_m", None)
    if sig is not None and sig > 0.0:
        return float(sig)
    return float(twin.arc_sigma_m)


def arc_pressure_sigma_cells(twin: "WAAMTwin") -> float:
    return arc_pressure_sigma_m(twin) / twin.grid.dx


def arc_pressure_peak_pa(twin: "WAAMTwin") -> float:
    """
    Resolve peak arc pressure for the current step.

    Models (twin.arc_pressure_model):
      - lin_eagar — Lin & Eagar I² / σ_p² (default for physics_tier: full)
      - constant  — twin.arc_pressure [Pa] (legacy / calibration)
    """
    model = str(getattr(twin, "arc_pressure_model", "constant")).strip().lower()
    if model in ("lin_eagar", "lineagar", "lin-eagar"):
        return lin_eagar_peak_pa(twin.welding_current_A, arc_pressure_sigma_m(twin))
    return float(twin.arc_pressure)


def droplet_impact_velocity_m_s(twin: "WAAMTwin", m_drop_kg: float | None = None) -> float:
    """Impact speed from feed, gravity, and a simple pinch term."""
    g = 9.81
    drop_len = 3e-3
    if twin.droplet_freq > 0 and twin.wire_feed_m_s > 0:
        drop_len = twin.wire_feed_m_s / twin.droplet_freq
    if m_drop_kg is None:
        m_drop_kg = droplet_mass_kg(twin)
    v_feed = max(twin.wire_feed_m_s, 0.0)
    v_grav = math.sqrt(max(0.0, 2.0 * g * drop_len))
    # Crude wire-tip pinch / electromagnetic assist term.
    dia_ref = max(twin.wire_diameter_m, 1e-6)
    pinch = 0.35 * math.sqrt(max(twin.welding_current_A, 0.0) / 200.0) * math.sqrt(1.2e-3 / dia_ref)
    mode = infer_transfer_mode(twin)
    if mode == "globular":
        mode_scale = 0.78
    elif mode == "spray":
        mode_scale = 1.18
    elif mode == "pulsed":
        mode_scale = 1.05
    else:
        mode_scale = 1.0
    mass_term = 1.0
    if m_drop_kg > 0:
        mass_term = max(0.8, min(1.25, math.sqrt(droplet_mass_kg(twin) / m_drop_kg)))
    return min(5.0, mode_scale * max(v_feed, v_grav + pinch) * mass_term)


def droplet_velocity_lu(twin: "WAAMTwin", m_drop_kg: float | None = None) -> tuple[float, float, float]:
    """Droplet velocity in lattice units with travel direction and lead angle."""
    grid = twin.grid
    v = droplet_impact_velocity_m_s(twin, m_drop_kg)
    lead = math.radians(float(getattr(twin, "impact_lead_angle_deg", 0.0)))
    dir_x, dir_y, _ = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
    dxy = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if dxy < 1e-9:
        dir_x, dir_y, dxy = 1.0, 0.0, 1.0
    dir_x /= dxy
    dir_y /= dxy
    v_xy = max(twin.travel_speed_m_s, abs(v) * math.tan(lead))
    v_xy = min(v * 0.65, v_xy)
    v_z = -math.sqrt(max(v * v - v_xy * v_xy, v * v * 0.20))
    return (
        v_xy * dir_x * grid.dt / grid.dx,
        v_xy * dir_y * grid.dt / grid.dx,
        v_z * grid.dt / grid.dx,
    )


def droplet_impact_pressure_pa(twin: "WAAMTwin", m_drop_kg: float | None = None, drop_r_m: float | None = None) -> float:
    """Kinetic pressure scale ½ρv² from droplet mass and contact area."""
    m = droplet_mass_kg(twin) if m_drop_kg is None else m_drop_kg
    if m <= 0:
        return 0.0
    v = droplet_impact_velocity_m_s(twin, m)
    r = drop_r_m if drop_r_m is not None else max(twin.grid.dx * 1.5, twin.wire_diameter_m * 0.5)
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
            float(getattr(twin, "recoil_accommodation", 0.54)),
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
    else:
        from .forces import apply_vapor_recoil
        apply_vapor_recoil(
            g.Fz, g.T, g.phi, g.flags,
            arc_i, arc_j, arc_k, twin.sigma_cells,
            twin.recoil_pressure_pa, twin.mat.T_liquidus,
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
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
        g.FLAG_SOLID, g.FLAG_GAS,
    )


def apply_droplet_impact(
    twin: "WAAMTwin",
    g: "WAAMGrid",
    arc_i: float,
    arc_j: float,
    arc_k: float,
    drop_r: float,
    drop_mass_kg: float | None = None,
) -> None:
    if not twin.enable_deposition_momentum:
        return
    vx, vy, vz = droplet_velocity_lu(twin, drop_mass_kg)
    kwf = _kwf()
    kwf.feed_wire_momentum_impact(
        g.f_src, g.rho,
        g.ux, g.uy, g.uz, g.flags, g.f_l,
        arc_i, arc_j, arc_k, drop_r,
        vx, vy, vz,
        g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    if twin.enable_droplet_impact_pressure:
        kwf.apply_droplet_impact_pressure(
            g.Fz, g.flags, g.phi, g.f_l,
            arc_i, arc_j, arc_k, drop_r,
            droplet_impact_pressure_pa(twin, drop_mass_kg, drop_r * g.dx),
            g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


def solve_lorentz(twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
    if not twin.enable_lorentz:
        return
    g.ensure_lorentz_fields()
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

    # Warm start: keep the potential from the previous step (the geometry
    # changes slowly, so a few dozen iterations/step track the solution).
    # Zeroing φ every call forced a cold start each timestep, which a fixed
    # iteration budget could never converge — the J×B force came from a
    # potential ~40× off on even a 32³ grid.
    cold_start = not getattr(twin, "_lorentz_warm", False)
    if cold_start:
        kwf.elec_init_ground(g.phi_elec, g.flags, twin.nz_solid, g.FLAG_SOLID)
        twin._lorentz_warm = True

    # Jacobi solve with convergence monitoring: iterate in chunks and stop
    # when the relative L1 change drops below tolerance. Jacobi needs
    # O(N²·ln 1/tol) iterations from a cold start, so the first solve gets a
    # grid-scaled cap; warm-started steps use the configured budget.
    max_iters = max(int(twin.lorentz_jacobi_iters), 20)
    if cold_start:
        n_max = max(g.nx, g.ny, g.nz)
        max_iters = max(max_iters, 2 * n_max * n_max)
    tol = float(getattr(twin, "lorentz_jacobi_tol", 1e-4))
    check_every = 10
    converged = False
    iters_done = 0
    while iters_done < max_iters:
        for _ in range(min(check_every, max_iters - iters_done)):
            kwf.elec_jacobi_step(
                g.phi_elec, g.phi_elec_tmp, g.sigma_elec, g.elec_source, g.flags,
                g.dx, twin.lorentz_jacobi_omega,
                g.FLAG_GAS, g.FLAG_SOLID, twin.nz_solid, g.nx, g.ny, g.nz,
            )
            kwf.elec_l1_diff(g.phi_elec_tmp, g.phi_elec, g.elec_res_buf, g.elec_norm_buf)
            g.phi_elec.copy_from(g.phi_elec_tmp)
            iters_done += 1
        rel = float(g.elec_res_buf[None]) / max(float(g.elec_norm_buf[None]), 1e-30)
        if rel < tol:
            converged = True
            break
    if not converged:
        twin._lorentz_unconverged = getattr(twin, "_lorentz_unconverged", 0) + 1
        twin._lorentz_unconverged_streak = getattr(twin, "_lorentz_unconverged_streak", 0) + 1
        if twin._lorentz_unconverged in (1, 10, 100) or twin._lorentz_unconverged % 1000 == 0:
            print(
                f"[weld_forces] WARNING: Lorentz Jacobi hit {max_iters} iters "
                f"without converging (rel Δ={rel:.2e} > {tol:.0e}); "
                f"raise lorentz_jacobi_iters. ({twin._lorentz_unconverged} occurrences)"
            )
    else:
        twin._lorentz_unconverged_streak = 0

    kwf.elec_compute_J(
        g.Jx, g.Jy, g.Jz, g.phi_elec, g.sigma_elec, g.flags, g.dx,
        g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    # Axisymmetric self-magnetic field from the enclosed axial current
    # (Ampère's law) — see kernels.elec_B_axisymmetric.
    bin_dr_cells = max(float(max(g.nx, g.ny)) / (2.0 * g.n_elec_bins), 0.5)
    kwf.elec_bin_axial_current(
        g.Jz, g.elec_rad_bins, arc_i, arc_j, g.dx,
        bin_dr_cells, g.n_elec_bins, g.nz,
    )
    kwf.elec_prefix_bins(g.elec_rad_bins, g.n_elec_bins, g.nz)
    kwf.elec_B_axisymmetric(
        g.Bx, g.By, g.Bz, g.elec_rad_bins,
        arc_i, arc_j, g.dx, bin_dr_cells, MU0, g.n_elec_bins, g.nz,
    )
    kwf.apply_lorentz_JxB(
        g.Fx, g.Fy, g.Fz,
        g.Jx, g.Jy, g.Jz, g.Bx, g.By, g.Bz,
        g.f_l, g.flags, g.dt, g.dx, twin.mat.rho, g.FLAG_GAS,
    )
