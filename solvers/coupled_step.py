"""
coupled_step.py — v2 simulation orchestration (single physics timestep order).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..physics import thermal, phase_change, forces, free_surface, lbm, deposition, weld_forces
from ..physics.electrical_stickout import droplet_entry_temperature_K, update_ctwd
from .. import kernels
from .. import logging_util as log

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid

# Curvature limit for the free-surface pressure condition (radius ≥ 2 cells).
FS_KAPPA_MAX = 0.5


def _clamp_enthalpy_ceiling(twin: "WAAMTwin", g) -> None:
    if not twin.enable_enthalpy_cap:
        return
    if getattr(twin, "use_dual_alloy", False):
        thermal.clamp_enthalpy_ceiling_dual(
            g.H, g.flags, g.alloy_frac,
            twin.H_cap, twin.plate_H_cap,
            g.dx, g.enthalpy_cap_energy_J_buf, g.FLAG_GAS,
            g.phi, twin.fs_iface_flag(),
        )
    elif twin.use_material_tables:
        thermal.clamp_enthalpy_ceiling_variable_cp(
            g.H, g.flags, twin.H_cap,
            g.dx, g.enthalpy_cap_energy_J_buf, g.FLAG_GAS,
            g.phi, twin.fs_iface_flag(),
        )
    else:
        thermal.clamp_enthalpy_ceiling_scalar(
            g.H, g.flags, twin.cp_rho,
            twin.mat.T_solidus, twin.mat.T_liquidus, twin.T_vapor_cap_K,
            twin.L_rho, g.dx, g.enthalpy_cap_energy_J_buf, g.FLAG_GAS,
            g.phi, twin.fs_iface_flag(),
        )
    e = float(g.enthalpy_cap_energy_J_buf[None])
    from ..physics.ledger import ensure_ledger
    ensure_ledger(twin).record_cap(e)
    twin._enthalpy_cap_energy_J_step = float(
        getattr(twin, "_enthalpy_cap_energy_J_step", 0.0)
    ) + e
    twin._enthalpy_cap_energy_J_cum = float(
        getattr(twin, "_enthalpy_cap_energy_J_cum", 0.0)
    ) + e


def _resolve_arc_k(
    twin: "WAAMTwin",
    g,
    arc_i: float,
    arc_j: float,
    torch_z_m: float | None = None,
) -> float:
    i0 = max(0, min(int(arc_i), g.nx - 1))
    j0 = max(0, min(int(arc_j), g.ny - 1))
    # Without VOF, wire feed or a window shift the surface never moves, so the
    # query (and its device sync) runs once per column.
    static_surface = (
        not twin.enable_vof
        and (twin.droplet_freq <= 0 or twin.wire_feed_m_s <= 0.0)
        and not twin.enable_moving_window
    )
    cache = twin.__dict__.setdefault("_surface_k_cache", {}) if static_surface else None
    k_surface = cache.get((i0, j0)) if cache is not None else None
    if k_surface is None:
        free_surface.surface_height_at(
            g.phi, g.flags, g.surface_k_buf,
            i0, j0, twin.nz_solid, g.FLAG_GAS, g.nz,
        )
        k_surface = float(g.surface_k_buf[None])
        if k_surface < 1.0:
            k_surface = float(max(1, twin.nz_solid - 1))
        if cache is not None:
            cache[(i0, j0)] = k_surface

    if getattr(twin, "use_torch_z", False) and torch_z_m is not None:
        # Two conventions for torch_z_m:
        #   1) Job / inline torch_path z_mm — height of the *bead build* above the
        #      plate top (layer height). Values are typically ≪ CTWD (mm–cm).
        #   2) Robot TCP Z — tip height above the substrate datum; bead top is
        #      TCP − CTWD. Values are typically ≥ CTWD (~10–20 mm).
        # Treating layer-height paths as TCP (subtracting CTWD) clamps the arc
        # to the plate forever and breaks multilayer deposition.
        ctwd = float(getattr(twin, "ctwd_m", 0.0) or 0.0)
        if ctwd > 1e-9 and float(torch_z_m) + 1e-12 >= ctwd:
            z_bead_m = float(torch_z_m) - ctwd
        else:
            z_bead_m = float(torch_z_m)
        z_bead_m = max(float(getattr(twin, "substrate_z_m", 0.0)), z_bead_m)
        # z = 0 is the plate-top face (index nz_solid), not the last metal cell.
        # A positive layer height lifts the arc by that many cells above the face.
        k_robot = twin.nz_solid + (z_bead_m - twin.substrate_z_m) / g.dx
        k_robot = max(float(twin.nz_solid), min(k_robot, float(g.nz) - 2.0))
        return max(k_surface, k_robot)

    return k_surface


def _solidify_if_enabled(twin: "WAAMTwin", g) -> None:
    if not twin.enable_substrate_growth and not twin.enable_bead_freeze:
        return
    if getattr(twin, "use_dual_alloy", False):
        free_surface.solidify_cooled_metal_dual(
            g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz, g.alloy_frac,
            twin.H_sol, twin.mat.T_solidus,
            twin.plate_H_sol, twin.plate_mat.T_solidus,
            twin.enable_bead_freeze or twin.enable_substrate_growth,
            g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
        )
        return
    free_surface.solidify_cooled_metal(
        g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz,
        twin.H_sol, twin.mat.T_solidus,
        twin.enable_bead_freeze or twin.enable_substrate_growth,
        g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
    )


def _refresh_thermal(twin: "WAAMTwin", g) -> bool:
    """Fill per-cell cp/α/dγ/τ. Returns True when the field thermal path is active."""
    if getattr(twin, "use_dual_alloy", False):
        wt = twin.gpu_tables
        pt = twin.gpu_tables_plate
        thermal.refresh_properties_dual(
            g.T, g.cp_rho_field, g.alpha_lu_field, g.dgamma_lu_field, g.tau_field,
            g.alloy_frac, g.flags,
            wt.cp_T, wt.cp_V, wt.k_T, wt.k_V, wt.mu_T, wt.mu_V,
            wt.dgamma_T, wt.dgamma_V, wt.n_cp, wt.n_k, wt.n_mu, wt.n_dgamma,
            pt.cp_T, pt.cp_V, pt.k_T, pt.k_V, pt.mu_T, pt.mu_V,
            pt.dgamma_T, pt.dgamma_V, pt.n_cp, pt.n_k, pt.n_mu, pt.n_dgamma,
            g.dt, g.dx,
            twin.mat.rho, twin.plate_mat.rho,
            wt.cp_fallback, wt.k_fallback, wt.mu_fallback, wt.dgamma_fallback,
            pt.cp_fallback, pt.k_fallback, pt.mu_fallback, pt.dgamma_fallback,
            twin.force_scale / g.dx, twin.plate_force_scale / g.dx,
            twin.cp_rho, twin.plate_cp_rho,
            twin.alpha_lu, twin.plate_alpha_lu,
            twin.dgamma_dT_lu, twin.plate_dgamma_dT_lu,
            g.tau, twin.plate_tau,
            twin.marangoni_scale,
            1 if wt.enabled else 0,
            1 if pt.enabled else 0,
            g.FLAG_GAS,
        )
        return True
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
            twin.force_scale / g.dx,
            twin.cp_rho, twin.alpha_lu, twin.dgamma_dT_lu, g.tau,
            twin.marangoni_scale,
            1, g.flags, g.FLAG_SOLID, g.FLAG_GAS,
        )
        return True
    return False


def _update_phase(twin: "WAAMTwin", g) -> None:
    if getattr(twin, "use_dual_alloy", False):
        wt = twin.gpu_tables
        pt = twin.gpu_tables_plate
        thermal.update_phase_dual(
            g.H, g.T, g.f_l, g.alloy_frac,
            twin.L_rho, twin.mat.T_solidus, twin.mat.T_liquidus,
            twin.H_sol, twin.H_liq,
            twin.plate_L_rho, twin.plate_mat.T_solidus, twin.plate_mat.T_liquidus,
            twin.plate_H_sol, twin.plate_H_liq,
            twin.mat.rho, twin.plate_mat.rho,
            wt.cp_T, wt.cp_V, wt.cp_H, wt.n_cp,
            pt.cp_T, pt.cp_V, pt.cp_H, pt.n_cp,
        )
        return
    if twin.use_material_tables:
        tbl = twin.gpu_tables
        thermal.update_phase_variable_cp(
            g.H, g.T, g.f_l,
            twin.L_rho,
            twin.mat.T_solidus, twin.mat.T_liquidus,
            twin.H_sol, twin.H_liq,
            twin.mat.rho,
            tbl.cp_T, tbl.cp_V, tbl.cp_H, tbl.n_cp,
        )
        return
    phase_change.update_phase(
        g.H, g.T, g.f_l,
        twin.cp_rho, twin.L_rho,
        twin.mat.T_solidus, twin.mat.T_liquidus,
    )


def _inject_arc(twin: "WAAMTwin", g, arc_i: float, arc_j: float, arc_k: float) -> None:
    """Arc heat for one step, booked in the ledger, then the enthalpy ceiling."""
    from ..physics.ledger import ensure_ledger
    from ..physics.arc import _arc_power_W

    twin._last_goldak_deposited_J = None
    twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)
    # Ledger arc energy actually injected (Phase 4 split-aware).
    # Absolute Goldak sets _last_goldak_deposited_J to Σ(ΔH)·dx³.
    deposited = getattr(twin, "_last_goldak_deposited_J", None)
    if deposited is not None:
        ensure_ledger(twin).record_arc(float(deposited))
    else:
        Q_arc, eta_arc = _arc_power_W(twin)
        ensure_ledger(twin).record_arc(float(eta_arc) * float(Q_arc) * float(g.dt))
    _clamp_enthalpy_ceiling(twin, g)


def _conduct_and_recover(twin: "WAAMTwin", g, use_fields: bool) -> None:
    """Advection–diffusion, surface losses, evaporation and T/f_l recovery."""
    # M1: snapshot H before advection so neighbour reads are race-free.
    g.H_tmp.copy_from(g.H)
    fs_iface = twin.fs_iface_flag()
    if use_fields:
        thermal.advect_diffuse_variable(
            g.H, g.H_tmp, g.T, g.ux, g.uy, g.uz, g.flags,
            g.alpha_lu_field, g.cp_rho_field, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz, g.phi, fs_iface,
        )
    else:
        thermal.advect_diffuse_temperature(
            g.H, g.H_tmp, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz, g.phi, fs_iface,
        )

    if twin.enable_heat_loss:
        t_cap = float(twin.T_vapor_cap_K) if twin.enable_enthalpy_cap else 0.0
        if use_fields:
            thermal.apply_boundary_losses_variable(
                g.H, g.T, g.flags, g.cp_rho_field,
                twin.T_amb,
                twin.h_conv, twin.eps_rad,
                1 if twin.enable_convection else 0,
                1 if twin.enable_radiation else 0,
                g.dt, g.dx, twin.sigma_sb, t_cap,
                g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz, g.energy_acc_J, g.phi, fs_iface,
            )
            thermal.clamp_enthalpy_floor(
                g.H, g.flags, g.alloy_frac,
                twin.H_amb, twin.plate_H_amb, g.FLAG_GAS, g.dx, g.energy_acc_J,
                g.phi, fs_iface,
            )
        else:
            thermal.apply_boundary_losses(
                g.H, g.T, g.flags,
                twin.T_amb,
                twin.h_conv, twin.eps_rad,
                1 if twin.enable_convection else 0,
                1 if twin.enable_radiation else 0,
                twin.cp_rho, g.dt, g.dx, twin.sigma_sb, t_cap,
                g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz, g.energy_acc_J, g.phi, fs_iface,
            )
            thermal.clamp_enthalpy_floor_scalar(
                g.H, twin.cp_rho, g.flags, twin.T_amb, g.FLAG_GAS, g.dx, g.energy_acc_J,
                g.phi, fs_iface,
            )

    _update_phase(twin, g)

    # Evaporative energy sink on the free surface (before hard vapor ceiling).
    # Onset must match CC recoil, including twin.T_recoil_onset_K if set.
    if getattr(twin, "enable_evaporative_cooling", False):
        T_boil = float(twin.T_boiling_K)
        T_onset = weld_forces.recoil_onset_K(twin)
        use_field = 1 if use_fields else 0
        thermal.apply_evaporative_enthalpy_sink(
            g.H, g.T, g.phi, g.f_l, g.flags,
            g.cp_rho_field,
            use_field,
            float(twin.cp_rho),
            T_boil,
            T_onset,
            float(twin.L_vapor_J_kg),
            float(twin.R_spec_vapor_J_kgK),
            float(twin.P_vapor_ref_Pa),
            float(getattr(twin, "recoil_accommodation", 0.54)),
            float(getattr(twin, "evap_cooling_scale", 1.0)),
            g.dt, g.dx,
            g.evap_energy_J_buf,
            g.FLAG_GAS,
            g.nx, g.ny, g.nz,
            fs_iface,
        )
        e_step = float(g.evap_energy_J_buf[None])
        twin._evap_energy_J_step = e_step
        twin._evap_energy_J_cum = float(getattr(twin, "_evap_energy_J_cum", 0.0)) + e_step
        from ..physics.ledger import ensure_ledger
        ensure_ledger(twin).record_evap(e_step)
        if not twin.enable_enthalpy_cap:
            _update_phase(twin, g)
    else:
        twin._evap_energy_J_step = 0.0

    # Cap H, then re-recover T so telemetry / T_max cannot retain a
    # post-phase spike above T_vapor_cap while H was already clamped.
    if twin.enable_enthalpy_cap:
        _clamp_enthalpy_ceiling(twin, g)
        _update_phase(twin, g)


def _refresh_lbm_band(twin: "WAAMTwin", g: "WAAMGrid", band: int) -> None:
    """Rebuild the streamed-gas mask (metal dilated by ``band`` cells).

    Rebuilt every band//2 steps (the interface moves < 0.1 cell/step under the
    Mach limit) and whenever a droplet fires or the window shifts, since both
    create metal away from the previous band.
    """
    g.ensure_lbm_band_fields()
    key = (
        int(getattr(twin, "_n_droplets_fired", 0)),
        float(getattr(twin, "_window_offset_x_m", 0.0)),
        float(getattr(twin, "_window_offset_y_m", 0.0)),
        float(getattr(twin, "_window_offset_z_m", 0.0)),
        band,
    )
    last = getattr(twin, "_lbm_band_state", None)
    if last is not None and last[0] == key and twin._step_n - last[1] < max(1, band // 2):
        return
    kernels.mark_non_gas(g.flags, g.lbm_active, g.FLAG_GAS)
    kernels.dilate_mask_axis(g.lbm_active, g.lbm_band_tmp, band, 0, g.nx)
    kernels.dilate_mask_axis(g.lbm_band_tmp, g.lbm_active, band, 1, g.ny)
    kernels.dilate_mask_axis(g.lbm_active, g.lbm_band_tmp, band, 2, g.nz)
    g.lbm_active.copy_from(g.lbm_band_tmp)
    twin._lbm_band_state = (key, twin._step_n)


def _free_surface_stream(twin: "WAAMTwin", g: "WAAMGrid") -> None:
    """Free-surface LBM stream and interface update (see kernels/fslbm.py)."""
    nx, ny, nz = g.nx, g.ny, g.nz
    gamma = float(twin.gamma_lu) if twin.enable_csf_tension else 0.0
    kernels.fs_prepare(g.flags, g.FLAG_FLUID, g.FLAG_GAS, g.FLAG_IFACE, nx, ny, nz)
    kernels.fs_stream(
        g.f_dst, g.f_src, g.rho, g.ux, g.uy, g.uz, g.phi, g.flags,
        g.fs_mass, g.fs_mark, g.H, g.fs_energy, gamma, FS_KAPPA_MAX,
        g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_IFACE, nx, ny, nz,
    )
    kernels.fs_mark_neighbors(g.flags, g.fs_mark, g.FLAG_GAS, nx, ny, nz)
    kernels.fs_init_new_iface(
        g.f_src, g.rho, g.ux, g.uy, g.uz, g.flags, g.fs_mark, g.fs_mass,
        g.fs_energy, g.H, g.T, g.f_l,
        g.FLAG_FLUID, g.FLAG_IFACE, nx, ny, nz,
    )
    kernels.fs_promote_new_iface(g.flags, g.fs_mark, g.fs_mass, g.rho, g.FLAG_IFACE)
    kernels.fs_distribute_excess(
        g.flags, g.fs_mark, g.fs_mass, g.rho, g.fs_energy, g.f_l, g.dx, g.energy_acc_J,
        g.fs_mass_lost, g.FLAG_FLUID, g.FLAG_GAS, g.FLAG_IFACE, nx, ny, nz,
    )
    kernels.fs_count_surface(
        g.flags, g.fs_mass_lost, g.fs_n_surface, g.fs_share,
        g.FLAG_GAS, g.FLAG_IFACE, nx, ny, nz,
    )
    kernels.fs_settle_unplaced(
        g.flags, g.fs_mass, g.fs_energy, g.H, g.fs_share, g.dx, g.energy_acc_J,
        g.FLAG_GAS, g.FLAG_IFACE, nx, ny, nz,
    )
    kernels.fs_update_phi(
        g.flags, g.phi, g.fs_mass, g.rho, g.fs_mark,
        g.FLAG_FLUID, g.FLAG_GAS, g.FLAG_IFACE,
    )
    kernels.fs_apply_energy(
        g.H, g.flags, g.phi, g.fs_energy, g.dx, g.energy_acc_J,
        g.FLAG_SOLID, g.FLAG_GAS,
    )


def _update_time_above(twin: "WAAMTwin", g) -> None:
    if getattr(twin, "use_dual_alloy", False):
        kernels.update_time_above_T_dual(
            g.T, g.flags, g.alloy_frac,
            g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
            g.dt, 800.0 + 273.15, 1100.0 + 273.15,
            twin.mat.T_solidus, twin.plate_mat.T_solidus,
            g.FLAG_GAS,
        )
    else:
        kernels.update_time_above_T(
            g.T, g.flags,
            g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
            g.dt, 800.0 + 273.15, 1100.0 + 273.15, twin.mat.T_solidus,
            g.FLAG_GAS,
        )


def coupled_step(
    twin: "WAAMTwin",
    torch_x_m: float,
    torch_y_m: float,
    is_welding: bool,
    torch_z_m: float | None = None,
) -> None:
    """One coupled physics timestep.

    Operator-splitting note: the thermal advection-diffusion step uses the
    velocity field from the END of the previous step (first-order Lie
    splitting). The flow⇄thermal coupling therefore lags by one Δt; the
    splitting error is O(Δt·|∂u/∂t|) and grows with Marangoni number. At the
    production Δt (≈ dx·u_lu/u_ref ~ 10 µs) this is far below the spatial
    discretization error, but it is a systematic phase lag — halve dt to
    check sensitivity if you operate at extreme surface-tension gradients.
    """
    if twin.thermal_fast_path:
        from .thermal_only import thermal_tier_step
        thermal_tier_step(twin, torch_x_m, torch_y_m, is_welding, torch_z_m)
        return

    g = twin.grid

    arc_i = torch_x_m / g.dx
    arc_j = torch_y_m / g.dx
    arc_k = _resolve_arc_k(twin, g, arc_i, arc_j, torch_z_m)
    twin._last_arc_ijk = (float(arc_i), float(arc_j), float(arc_k))

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    twin._enthalpy_cap_energy_J_step = 0.0
    # Droplet booking from *previous* step is consumed by _arc_power_W when
    # energy_split is on; clear after inject so a no-droplet step restores full ηQ.
    twin._droplet_enthalpy_booked_J_prev = float(
        getattr(twin, "_droplet_enthalpy_booked_J_step", 0.0)
    )
    twin._droplet_enthalpy_booked_J_step = 0.0

    if twin.enable_ctwd:
        update_ctwd(twin, g)

    # H6: fill cp/α/dγ/τ fields BEFORE any enthalpy ceiling so the cap uses
    # real ρ·cp (previously cp_rho_field was 0 → step-1 wiped the plate).
    use_fields = _refresh_thermal(twin, g)

    # Peak arc pressure: Lin–Eagar I²/σ² or constant Pa (see weld_forces).
    current_pressure = weld_forces.arc_pressure_peak_pa(twin)
    pressure_sigma_cells = weld_forces.arc_pressure_sigma_cells(twin)

    deposited_this_step = False
    if is_welding:
        _inject_arc(twin, g, arc_i, arc_j, arc_k)

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
                g.deposit_real_buf[None] = 0.0
                foot_r = deposition.deposition_footprint_cells(twin, drop_r)
                r_try = foot_r
                # Historical cap was 16 cells at dx=0.4 mm (~6.4 mm physical).
                # A fixed cell cap shrinks the physical footprint on fine meshes and
                # causes deposition overflow / strict mass_balance failures (HPC dx=0.2).
                max_foot_mm = float(
                    getattr(twin, "deposition_max_footprint_mm", 6.4) or 6.4
                )
                max_r_cells = max(16.0, max_foot_mm / max(g.dx * 1000.0, 1e-9))
                placed_real = 0.0
                # M7: H = cp·T_sol + L + cp·(T_drop − T_liq). The kernel uses
                # H = cp·T_drop + L_eff, so L_eff = L − cp·(T_liq − T_sol).
                dT_mush = float(twin.mat.T_liquidus - twin.mat.T_solidus)
                L_eff = float(twin.L_rho) - float(twin.cp_rho) * dT_mush
                for _ in range(16):
                    # Resume budget from metal already placed so retries expand
                    # the footprint instead of re-filling with thermal no-ops.
                    g.deposit_vol_buf[None] = placed_real
                    g.deposit_real_buf[None] = placed_real
                    if twin.fs_iface_flag() >= 0:
                        kernels.fs_deposit_into_interface(
                            g.phi, g.flags, g.H,
                            arc_i, arc_j, arc_k, r_try, drop_r, drop_vol,
                            float(twin.cp_rho) * T_drop + L_eff,
                            g.deposit_vol_buf, g.deposit_real_buf, g.dx ** 3,
                            g.FLAG_IFACE, g.nx, g.ny, g.nz,
                        )
                    deposition.feed_wire_surface(
                        g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
                        arc_i, arc_j, arc_k,
                        r_try, drop_r,
                        drop_vol,
                        T_drop,
                        twin.cp_rho, L_eff,
                        1.0,  # lattice density (rho_lu ≈ 1 convention)
                        g.deposit_vol_buf, g.deposit_real_buf, g.dx ** 3,
                        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                        g.nx, g.ny, g.nz,
                    )
                    placed_real = float(g.deposit_real_buf[None])
                    if placed_real >= drop_vol * 0.98:
                        break
                    r_try = min(r_try + 1.5, max_r_cells)
                # Mass ledger counts only real gas→fluid conversions.
                if placed_real < drop_vol * 0.98:
                    twin._deposition_overflow += 1
                    if not twin._warned_overflow and twin._deposition_overflow >= 20:
                        twin._warned_overflow = True
                        log.warning(
                            f"[coupled_step] WARNING: {twin._deposition_overflow} droplets "
                            f"deposited less volume than the wire feed supplies "
                            f"(latest: {placed_real / max(drop_vol, 1e-30):.0%} of target). "
                            f"Cumulative mass balance will drift below 1.0 — check "
                            f"mass_balance_ratio in telemetry."
                        )
                twin._deposited_volume_m3 += placed_real
                twin._n_droplets_fired += 1
                deposited_this_step = True
                # Ledger droplet enthalpy for converted volume
                from ..physics.ledger import ensure_ledger
                H_drop = (
                    float(twin.cp_rho) * float(twin.mat.T_solidus)
                    + float(twin.L_rho)
                    + float(twin.cp_rho) * (float(T_drop) - float(twin.mat.T_liquidus))
                )
                drop_H_J = H_drop * placed_real
                twin._droplet_enthalpy_booked_J_step = float(drop_H_J)
                ensure_ledger(twin).record_droplet_enthalpy(drop_H_J)
                weld_forces.apply_droplet_impact(twin, g, arc_i, arc_j, arc_k, drop_r, drop_mass)

    # Properties already refreshed at step start (H6); refresh again after
    # deposition so newly created metal cells get correct tables.
    if deposited_this_step:
        use_fields = _refresh_thermal(twin, g)
    _conduct_and_recover(twin, g, use_fields)

    if (
        getattr(twin, "enable_alloy_mixing", False)
        and getattr(twin, "use_dual_alloy", False)
    ):
        kernels.mix_alloy_fusion_zone(
            g.alloy_frac, g.alloy_frac_buf, g.f_l, g.flags,
            float(getattr(twin, "alloy_mix_rate", 0.25)),
            g.FLAG_GAS, g.nx, g.ny, g.nz,
        )

    thermal.update_T_max(g.T, g.T_max, g.flags, g.FLAG_GAS)

    thermal.update_cooling_rate(
        g.T, g.T_prev, g.dT_dt, g.flags, g.dt, g.FLAG_GAS,
    )

    fs_on = bool(twin.enable_vof and getattr(twin, "use_free_surface_lbm", False))
    if fs_on:
        g.ensure_fslbm_fields()
        kernels.fs_update_phase_flags(
            g.phi, g.f_l, g.flags, g.f_src, g.rho, g.ux, g.uy, g.uz,
            twin.nz_solid, g.FLAG_FLUID, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_IFACE,
        )
    elif twin.enable_vof:
        g.ensure_vof_buffers()
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
            g.H, g.dx, g.energy_acc_J,
        )
        # Clear contradictory liquid-in-gas / solid-with-f_l after flag update.
        free_surface.sync_phi_liquid_fraction(
            g.phi, g.f_l, g.flags, g.FLAG_GAS, g.FLAG_SOLID,
        )

    # ── Force assembly (PHYSICS_FORCE_CORRECTNESS_SPEC §4.1) ─────────────
    # clear_forces already ran at step start. Every kernel below MUST use +=
    # only. Droplet impact pressure (if any) was applied earlier this step
    # during deposition and must be preserved by additive surface forces.
    #
    # Order: CSF → Marangoni → gas shear → arc pressure → recoil
    #      → hydrostatic → Boussinesq → Lorentz → collide (Darcy inside).

    # Under the free-surface LBM, surface tension enters through the
    # interface pressure condition instead of a CSF body force.
    if twin.enable_csf_tension and not fs_on:
        forces.compute_csf_tension(
            g.phi, g.flags, g.Fx, g.Fy, g.Fz,
            twin.gamma_lu,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
            enable_wetting=twin.enable_wetting,
            theta_rad=twin.theta_rad,
        )

    if getattr(twin, "enable_marangoni", True):
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

    if is_welding and twin.enable_gas_shear:
        weld_forces.apply_gas_shear(twin, g, arc_i, arc_j, arc_k)

    if is_welding and getattr(twin, "enable_arc_pressure", True):
        alloy_id, rho_w, rho_p = weld_forces.alloy_rho_args(twin)
        forces.apply_arc_pressure(
            g.Fz, g.flags, g.phi,
            arc_i, arc_j, arc_k, pressure_sigma_cells,
            current_pressure, g.dt, g.dx, alloy_id, rho_w, rho_p,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
    if is_welding:
        weld_forces.apply_recoil(twin, g, arc_i, arc_j, arc_k)

    # rho_ref = 1.0: lattice force densities under the ρ_lu ≈ 1 convention.
    if twin.enable_hydrostatic_gravity:
        forces.add_hydrostatic_gravity(
            g.Fz, g.f_l, g.flags, 1.0, twin.g_lu,
            g.FLAG_SOLID, g.FLAG_GAS,
        )

    if getattr(twin, "enable_buoyancy", True):
        forces.add_buoyancy(
            g.T, g.Fz, g.f_l, g.flags,
            twin.g_lu, twin.beta_T,
            twin.mat.T_liquidus,
            1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
        )

    if is_welding and twin.enable_lorentz:
        from ..physics.efficiency import lorentz_due
        if lorentz_due(twin):
            if getattr(twin, "lorentz_multigrid", False):
                from ..physics.lorentz_mg import solve_lorentz_multigrid
                solve_lorentz_multigrid(twin, g, arc_i, arc_j, arc_k)
            else:
                weld_forces.solve_lorentz(twin, g, arc_i, arc_j, arc_k)
        else:
            weld_forces.apply_cached_lorentz(twin, g)

    # Stability: full-tier surface/body forces on coarse grids can drive Ma≫1.
    # Prefer physical force_limit_m_s2 (mesh-scaled LU) when set — Phase 3 / Level A.
    # Hit counts are surfaced in telemetry. strict_mode warns once if they persist.
    u_cap = float(getattr(twin, "u_mach_limit_lu", 0.08))
    if hasattr(twin, "effective_force_limit_lu"):
        F_cap = float(twin.effective_force_limit_lu())
    else:
        F_cap = float(getattr(twin, "force_limit_lu", 0.05))
    twin._force_clamp_hits_step = 0
    twin._mach_clamp_hits_step = 0
    if F_cap > 0.0:
        kernels.clamp_body_force_magnitude(
            g.Fx, g.Fy, g.Fz, g.flags, F_cap,
            g.clamp_force_hits_buf,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
        n_f = int(g.clamp_force_hits_buf[None])
        twin._force_clamp_hits_step = n_f
        twin._force_clamp_hits_cum = int(getattr(twin, "_force_clamp_hits_cum", 0)) + n_f
        twin._force_clamp_steps = int(getattr(twin, "_force_clamp_steps", 0)) + (1 if n_f > 0 else 0)
        if n_f > 0 and getattr(twin, "warn_on_force_clamp", False):
            from .. import logging_util as log
            a_phys = float(getattr(twin, "force_limit_m_s2", 0.0) or 0.0)
            extra = f", a_phys={a_phys:g} m/s²" if a_phys > 0.0 else ""
            log.warning(
                f"[coupled_step] body-force clamp hit {n_f} cells this step "
                f"(cap={F_cap:g} lu/ts²{extra})"
            )

    C_darcy_eff = float(twin.effective_C_darcy())
    if twin.use_material_tables and twin.use_variable_tau and (not twin.use_srt):
        # TRT (two-rate) with per-cell μ(T) → τ; magic Λ=1/4 for odd modes.
        lbm.collide_mrt_variable_tau(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.tau_field, g.flags,
            g.ex, g.ey, g.ez, g.w, g.opp,
            twin.omega_bulk,
            C_darcy_eff,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
            float(getattr(twin, "smagorinsky_cs", 0.0) or 0.0),
        )
    elif twin.use_material_tables and twin.use_variable_tau:
        lbm.collide_srt_variable_tau(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.tau_field, C_darcy_eff,
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
            C_darcy_eff,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
    else:
        lbm.collide_mrt(
            g.f_src, g.f_dst,
            g.rho, g.ux, g.uy, g.uz,
            g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags,
            g.ex, g.ey, g.ez, g.w, g.opp,
            twin.omega, twin.omega_bulk,
            C_darcy_eff,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )

    if u_cap > 0.0:
        kernels.clamp_velocity_mach(
            g.f_dst, g.rho, g.ux, g.uy, g.uz, g.flags, u_cap,
            g.clamp_mach_hits_buf,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
        n_u = int(g.clamp_mach_hits_buf[None])
        twin._mach_clamp_hits_step = n_u
        twin._mach_clamp_hits_cum = int(getattr(twin, "_mach_clamp_hits_cum", 0)) + n_u
        twin._mach_clamp_steps = int(getattr(twin, "_mach_clamp_steps", 0)) + (1 if n_u > 0 else 0)

    band = int(getattr(twin, "lbm_gas_band", 0) or 0)
    if fs_on:
        _free_surface_stream(twin, g)
    elif band > 0 and not getattr(twin, "periodic_x", False):
        _refresh_lbm_band(twin, g, band)
        kernels.stream_banded(
            g.f_dst, g.f_src, g.flags, g.lbm_active,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    else:
        lbm.stream(
            g.f_dst, g.f_src,
            g.flags,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
            1 if getattr(twin, "periodic_x", False) else 0,
        )

    kernels.advect_tracers(
        g.porosity_pos, g.porosity_active,
        g.ux, g.uy, g.uz, g.f_l, g.flags,
        g.dx, g.dt, g.max_tracers,
        g.FLAG_SOLID, g.FLAG_GAS,
    )

    _update_time_above(twin, g)

    if twin.enable_substrate_growth or twin.enable_bead_freeze:
        if getattr(twin, "use_dual_alloy", False):
            free_surface.remelt_hot_solid_dual(
                g.T, g.H, g.f_l, g.phi, g.flags, g.alloy_frac,
                g.f_src, g.rho, g.ux, g.uy, g.uz, g.w,
                twin.L_rho, twin.H_sol, twin.H_liq,
                twin.plate_L_rho, twin.plate_H_sol, twin.plate_H_liq,
                g.FLAG_SOLID, g.FLAG_FLUID,
            )
        elif twin.use_material_tables:
            free_surface.remelt_hot_solid(
                g.T, g.H, g.f_l, g.phi, g.flags,
                g.f_src, g.rho, g.ux, g.uy, g.uz, g.w,
                twin.L_rho, twin.H_sol, twin.H_liq,
                g.FLAG_SOLID, g.FLAG_FLUID,
            )
        else:
            free_surface.remelt_hot_solid_scalar(
                g.T, g.H, g.f_l, g.phi, g.flags,
                g.f_src, g.rho, g.ux, g.uy, g.uz, g.w,
                twin.L_rho, twin.H_sol, twin.H_liq,
                g.FLAG_SOLID, g.FLAG_FLUID,
            )
        # H3: trailing solidify is a geometric prior that destroys energy.
        # Disabled by default; enable only for Tier-C heuristic legacy runs.
        if (
            is_welding
            and twin.enable_bead_freeze
            and getattr(twin, "enable_trailing_solidify", False)
        ):
            dir_x, dir_y, dir_z = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
            lookback_cells = max(2.0, twin.trailing_solidify_lookback_mm / (g.dx * 1000.0))
            T_freeze = twin.mat.T_liquidus + twin.trailing_solidify_temp_margin_K
            if getattr(twin, "use_dual_alloy", False):
                T_freeze_p = twin.plate_mat.T_liquidus + twin.trailing_solidify_temp_margin_K
                kernels.solidify_trailing_pool_dual(
                    g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz,
                    g.cp_rho_field, g.alloy_frac,
                    arc_i, arc_j, arc_k, dir_x, dir_y, dir_z, lookback_cells,
                    T_freeze, twin.mat.T_solidus,
                    T_freeze_p, twin.plate_mat.T_solidus,
                    g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
                )
            elif twin.use_material_tables:
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

    # Collide wrote f_dst and stream pulled it back into f_src, so f_src already
    # holds the next state; swapping here would feed the unstreamed buffer to
    # the next collide and discard propagation.
    twin._step_n += 1

    if hasattr(twin, "probe_recorder") and twin.probe_recorder is not None:
        twin.probe_recorder.record_step(twin)
