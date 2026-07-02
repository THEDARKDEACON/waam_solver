"""
coupled_step.py — v2 simulation orchestration (single physics timestep order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..physics import thermal, phase_change, forces, free_surface, lbm, deposition, weld_forces
from ..physics.electrical_stickout import droplet_entry_temperature_K, update_ctwd
from .. import kernels

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def _clamp_enthalpy_ceiling(twin: "WAAMTwin", g) -> None:
    if not twin.enable_enthalpy_cap:
        return
    if twin.use_material_tables:
        thermal.clamp_enthalpy_ceiling_variable_cp(
            g.H, g.flags, g.cp_rho_field,
            twin.mat.T_liquidus, twin.T_vapor_cap_K, twin.L_rho,
            g.FLAG_GAS,
        )
    else:
        thermal.clamp_enthalpy_ceiling_scalar(
            g.H, g.flags, twin.cp_rho,
            twin.mat.T_solidus, twin.mat.T_liquidus, twin.T_vapor_cap_K,
            twin.L_rho, g.FLAG_GAS,
        )


def _resolve_arc_k(
    twin: "WAAMTwin",
    g,
    arc_i: float,
    arc_j: float,
    torch_z_m: float | None = None,
) -> float:
    i0 = max(0, min(int(arc_i), g.nx - 1))
    j0 = max(0, min(int(arc_j), g.ny - 1))
    free_surface.surface_height_at(
        g.phi, g.flags, g.surface_k_buf,
        i0, j0, twin.nz_solid, g.FLAG_GAS, g.nz,
    )
    k_surface = float(g.surface_k_buf[None])
    if k_surface < 1.0:
        k_surface = float(max(1, twin.nz_solid - 1))

    if getattr(twin, "use_torch_z", False) and torch_z_m is not None:
        # Robot Z + CTWD → local bead-top height above substrate datum.
        z_bead_m = torch_z_m - twin.ctwd_m
        z_bead_m = max(twin.substrate_z_m, z_bead_m)
        k_robot = twin.nz_solid + (z_bead_m - twin.substrate_z_m) / g.dx - 1.0
        k_robot = max(float(twin.nz_solid) + 0.5, min(k_robot, float(g.nz) - 2.0))
        return max(k_surface, k_robot)

    return k_surface


def _solidify_if_enabled(twin: "WAAMTwin", g) -> None:
    if not twin.enable_substrate_growth and not twin.enable_bead_freeze:
        return
    free_surface.solidify_cooled_metal(
        g.T, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz,
        twin.mat.T_solidus,
        twin.enable_bead_freeze or twin.enable_substrate_growth,
        g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
    )


def coupled_step(
    twin: "WAAMTwin",
    torch_x_m: float,
    torch_y_m: float,
    is_welding: bool,
    torch_z_m: float | None = None,
) -> None:
    g = twin.grid

    arc_i = torch_x_m / g.dx
    arc_j = torch_y_m / g.dx
    arc_k = _resolve_arc_k(twin, g, arc_i, arc_j, torch_z_m)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)

    if twin.enable_ctwd:
        update_ctwd(twin, g)

    current_pressure = twin.arc_pressure

    if is_welding:
        twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)
        _clamp_enthalpy_ceiling(twin, g)

        sim_time = twin._step_n * g.dt
        if twin.droplet_freq > 0:
            period = deposition.droplet_period_s(twin)
            if sim_time - twin._last_droplet_time >= period:
                drop_dt = sim_time - twin._last_droplet_time
                twin._last_droplet_time = sim_time
                drop_mass = deposition.droplet_mass_for_interval_kg(twin, drop_dt)
                drop_r = deposition.droplet_radius_cells_from_mass_kg(twin, drop_mass)
                drop_vol = drop_mass / twin.mat.rho
                T_drop = droplet_entry_temperature_K(twin)
                kernels.inject_tracers(
                    g.porosity_pos, g.porosity_active, g.tracer_head,
                    g.max_tracers,
                    torch_x_m, torch_y_m, float(arc_k + 1) * g.dx,
                    float(twin.sigma_cells * g.dx), 50,
                )
                g.deposit_vol_buf[None] = 0.0
                foot_r = deposition.deposition_footprint_cells(twin, drop_r)
                r_try = foot_r
                for _ in range(8):
                    deposition.feed_wire_surface(
                        g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
                        arc_i, arc_j, arc_k,
                        r_try, drop_r,
                        drop_vol,
                        T_drop,
                        twin.cp_rho, twin.L_rho, twin.mat.rho,
                        g.deposit_vol_buf, g.dx ** 3,
                        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                        g.nx, g.ny, g.nz,
                    )
                    if float(g.deposit_vol_buf[None]) >= drop_vol * 0.98:
                        break
                    r_try = min(r_try + 1.5, 14.0)
                placed = float(g.deposit_vol_buf[None])
                if placed < drop_vol * 0.98:
                    twin._deposition_overflow += 1
                twin._deposited_volume_m3 += placed
                twin._n_droplets_fired += 1
                weld_forces.apply_droplet_impact(twin, g, arc_i, arc_j, arc_k, drop_r, drop_mass)

    if twin.use_material_tables:
        tbl = twin.gpu_tables
        thermal.refresh_properties(
            g.T, g.cp_rho_field, g.alpha_lu_field, g.dgamma_lu_field, g.tau_field,
            tbl.cp_T, tbl.cp_V, tbl.k_T, tbl.k_V,
            tbl.mu_T, tbl.mu_V,
            tbl.dgamma_T, tbl.dgamma_V,
            tbl.n_cp, tbl.n_k, tbl.n_mu, tbl.n_dgamma,
            g.mat.rho, g.dt, g.dx,
            tbl.cp_fallback, tbl.k_fallback, tbl.mu_fallback, tbl.dgamma_fallback,
            twin.force_scale,
            twin.cp_rho, twin.alpha_lu, twin.dgamma_dT_lu, g.tau,
            twin.marangoni_scale,
            1, g.flags, g.FLAG_SOLID, g.FLAG_GAS,
        )
        thermal.advect_diffuse_variable(
            g.H, g.T, g.ux, g.uy, g.uz, g.flags,
            g.alpha_lu_field, g.cp_rho_field, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
    else:
        thermal.advect_diffuse_temperature(
            g.H, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz,
        )

    if twin.enable_heat_loss:
        if twin.use_material_tables:
            thermal.apply_boundary_losses_variable(
                g.H, g.T, g.flags, g.cp_rho_field,
                twin.T_amb,
                twin.h_conv, twin.eps_rad,
                1 if twin.enable_convection else 0,
                1 if twin.enable_radiation else 0,
                g.dt, g.dx, twin.sigma_sb,
                g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz,
            )
            thermal.clamp_enthalpy_floor(
                g.H, g.cp_rho_field, g.flags, twin.T_amb, g.FLAG_GAS,
            )
        else:
            thermal.apply_boundary_losses(
                g.H, g.T, g.flags,
                twin.T_amb,
                twin.h_conv, twin.eps_rad,
                1 if twin.enable_convection else 0,
                1 if twin.enable_radiation else 0,
                twin.cp_rho, g.dt, g.dx, twin.sigma_sb,
                g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz,
            )
            thermal.clamp_enthalpy_floor_scalar(
                g.H, twin.cp_rho, g.flags, twin.T_amb, g.FLAG_GAS,
            )

    thermal.update_T_max(g.T, g.T_max, g.flags, g.FLAG_GAS)

    if twin.use_material_tables:
        thermal.update_phase_variable_cp(
            g.H, g.T, g.f_l, g.cp_rho_field,
            twin.L_rho,
            twin.mat.T_solidus, twin.mat.T_liquidus,
            twin.H_sol, twin.H_liq,
        )
    else:
        phase_change.update_phase(
            g.H, g.T, g.f_l,
            twin.cp_rho, twin.L_rho,
            twin.mat.T_solidus, twin.mat.T_liquidus,
        )

    _clamp_enthalpy_ceiling(twin, g)

    thermal.update_cooling_rate(
        g.T, g.T_prev, g.dT_dt, g.flags, g.dt, g.FLAG_GAS,
    )

    if twin.enable_vof:
        free_surface.advect_phi(
            g.phi_tmp, g.phi, g.ux, g.uy, g.uz, g.flags,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        g.phi.copy_from(g.phi_tmp)
        free_surface.reinitialize_phi(
            g.phi, g.flags, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_FLUID,
        )
        if twin.enable_wetting:
            free_surface.apply_contact_angle_phi_bc(
                g.phi, g.flags, twin.theta_rad,
                g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz,
            )
        free_surface.update_flags_from_phi(
            g.phi, g.f_l, g.flags, twin.nz_solid,
            g.FLAG_FLUID, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_IFACE,
        )

    if twin.enable_csf_tension:
        forces.compute_csf_tension(
            g.phi, g.flags, g.Fx, g.Fy, g.Fz,
            twin.gamma_lu,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
            enable_wetting=twin.enable_wetting,
            theta_rad=twin.theta_rad,
        )

    if twin.use_material_tables:
        forces.compute_marangoni_force_variable(
            g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
            g.dgamma_lu_field,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
    else:
        forces.compute_marangoni_force(
            g.T, g.phi, g.f_l,
            g.Fx, g.Fy, g.Fz,
            g.flags,
            twin.dgamma_dT_lu, g.dx,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )

    if twin.enable_hydrostatic_gravity:
        forces.add_hydrostatic_gravity(
            g.Fz, g.f_l, g.flags, g.mat.rho, twin.g_lu,
            g.FLAG_SOLID, g.FLAG_GAS,
        )

    forces.add_buoyancy(
        g.T, g.Fz, g.f_l, g.flags,
        twin.g_lu, twin.beta_T,
        twin.mat.T_liquidus,
        g.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )

    if is_welding:
        if twin.enable_lorentz:
            weld_forces.solve_lorentz(twin, g, arc_i, arc_j, arc_k)
        if twin.enable_gas_shear:
            weld_forces.apply_gas_shear(twin, g, arc_i, arc_j, arc_k)

    if is_welding:
        forces.apply_arc_pressure(
            g.Fz, g.flags, g.phi,
            arc_i, arc_j, arc_k, twin.sigma_cells,
            current_pressure, g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
        weld_forces.apply_recoil(twin, g, arc_i, arc_j, arc_k)

    if twin.use_material_tables and twin.use_variable_tau:
        lbm.collide_srt_variable_tau(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.tau_field, twin.C_darcy,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
    elif twin.use_srt:
        lbm.collide_srt(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.tau, twin.omega, 1.0,
            twin.C_darcy,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
    else:
        omega_s = twin.omega
        lbm.collide_mrt(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.ex, g.ey, g.ez, g.w, g.opp,
            omega_s, omega_s,
            twin.C_darcy,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )

    lbm.stream(
        g.f_dst, g.f_src,
        g.flags,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )

    kernels.advect_tracers(
        g.porosity_pos, g.porosity_active,
        g.ux, g.uy, g.uz, g.f_l, g.flags,
        g.dx, g.dt, g.max_tracers,
        g.FLAG_SOLID, g.FLAG_GAS,
    )

    kernels.update_time_above_T(
        g.T, g.flags,
        g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
        g.dt, 800.0 + 273.15, 1100.0 + 273.15, twin.mat.T_solidus,
        g.FLAG_GAS,
    )
    kernels.snapshot_forces(g.Fx, g.Fy, g.Fz, g.Fx_snap, g.Fy_snap, g.Fz_snap)

    if twin.enable_substrate_growth or twin.enable_bead_freeze:
        if twin.enable_substrate_growth or twin.enable_bead_freeze:
            if twin.use_material_tables:
                free_surface.remelt_hot_solid(
                    g.T, g.H, g.f_l, g.phi, g.flags, g.cp_rho_field,
                    twin.mat.T_solidus, twin.mat.T_liquidus, twin.L_rho,
                    g.FLAG_SOLID, g.FLAG_FLUID,
                )
            else:
                free_surface.remelt_hot_solid_scalar(
                    g.T, g.H, g.f_l, g.phi, g.flags,
                    twin.cp_rho,
                    twin.mat.T_solidus, twin.mat.T_liquidus, twin.L_rho,
                    g.FLAG_SOLID, g.FLAG_FLUID,
                )
        if is_welding and twin.enable_bead_freeze:
            dir_x, dir_y, dir_z = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
            lookback_cells = max(2.0, twin.trailing_solidify_lookback_mm / (g.dx * 1000.0))
            T_freeze = twin.mat.T_liquidus + twin.trailing_solidify_temp_margin_K
            if twin.use_material_tables:
                kernels.solidify_trailing_pool(
                    g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz, g.cp_rho_field,
                    arc_i, arc_j, arc_k, dir_x, dir_y, dir_z, lookback_cells, T_freeze,
                    twin.mat.T_solidus,
                    g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
                )
            else:
                kernels.solidify_trailing_pool_scalar(
                    g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz, twin.cp_rho,
                    arc_i, arc_j, arc_k, dir_x, dir_y, dir_z, lookback_cells, T_freeze,
                    twin.mat.T_solidus,
                    g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
                )
        _solidify_if_enabled(twin, g)

    g.swap_buffers()
    twin._step_n += 1

    if hasattr(twin, "probe_recorder") and twin.probe_recorder is not None:
        twin.probe_recorder.record_step(twin)
