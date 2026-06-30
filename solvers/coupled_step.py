"""
coupled_step.py — v2 simulation orchestration (single physics timestep order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..physics import thermal, phase_change, forces, free_surface, lbm, deposition
from .. import kernels

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def _resolve_arc_k(twin: "WAAMTwin", g, arc_i: float, arc_j: float) -> float:
    i0 = max(0, min(int(arc_i), g.nx - 1))
    j0 = max(0, min(int(arc_j), g.ny - 1))
    free_surface.surface_height_at(
        g.phi, g.flags, g.surface_k_buf,
        i0, j0, twin.nz_solid, g.FLAG_GAS, g.nz,
    )
    k = float(g.surface_k_buf[None])
    if k < 1.0:
        k = float(max(1, twin.nz_solid - 1))
    return k


def coupled_step(twin: "WAAMTwin", torch_x_m: float, torch_y_m: float, is_welding: bool) -> None:
    g = twin.grid

    arc_i = torch_x_m / g.dx
    arc_j = torch_y_m / g.dx
    arc_k = _resolve_arc_k(twin, g, arc_i, arc_j)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)

    current_pressure = twin.arc_pressure

    if is_welding:
        twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)

        sim_time = twin._step_n * g.dt
        if twin.droplet_freq > 0:
            period = 1.0 / twin.droplet_freq
            if sim_time - twin._last_droplet_time >= period:
                current_pressure += 100_000.0
                twin._last_droplet_time = sim_time
                kernels.inject_tracers(
                    g.porosity_pos, g.porosity_active, g.tracer_head,
                    g.max_tracers,
                    torch_x_m, torch_y_m, float(arc_k + 1) * g.dx,
                    float(twin.sigma_cells * g.dx), 50,
                )
                deposition.feed_wire(
                    g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
                    arc_i, arc_j, arc_k,
                    3.0,
                    twin.mat.T_liquidus + 500.0,
                    twin.cp_rho, twin.L_rho, twin.mat.rho,
                    g.FLAG_GAS, g.FLAG_FLUID,
                    g.nx, g.ny, g.nz,
                )
                if twin.enable_deposition_momentum:
                    deposition.feed_wire_momentum(
                        g.ux, g.uy, g.uz, g.flags, g.f_l,
                        arc_i, arc_j, arc_k, 3.0,
                        twin.droplet_vz_lu,
                        g.FLAG_GAS, g.nx, g.ny, g.nz,
                    )

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

    thermal.update_cooling_rate(
        g.T, g.T_prev, g.dT_dt, g.flags, g.dt, g.FLAG_GAS,
    )

    if twin.enable_substrate_growth:
        free_surface.solidify_cooled_metal(
            g.T, g.f_l, g.phi, g.flags,
            twin.mat.T_solidus,
            g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
        )
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

    forces.add_buoyancy(
        g.T, g.Fz, g.f_l, g.flags,
        twin.g_lu, twin.beta_T,
        twin.mat.T_liquidus,
        g.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )

    if is_welding:
        forces.apply_arc_pressure(
            g.Fz, g.flags, g.phi,
            arc_i, arc_j, arc_k, twin.sigma_cells,
            current_pressure, g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
        if twin.enable_recoil:
            forces.apply_vapor_recoil(
                g.Fz, g.T, g.phi, g.flags,
                arc_i, arc_j, arc_k, twin.sigma_cells,
                twin.recoil_pressure_pa, twin.mat.T_liquidus,
                g.dt, g.dx, twin.mat.rho,
                g.FLAG_SOLID, g.FLAG_GAS,
            )

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

    g.swap_buffers()
    twin._step_n += 1
