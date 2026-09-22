"""
kernels — Pure Quadrants GPU kernels (package split).

ALL simulation physics live here as @ti.kernel functions.
No Python logic runs inside these — they compile to native GPU code.

Collision: two-rate central-moment MRT (not a Geier-style cumulant operator).
The second-order stress moments relax at τ; higher moments use a separate
rate. See README collision honesty note. Offline derivation of the FMA
sequence lives in tools/derive_cumulant.py.

NOTE: This package must be imported AFTER init_taichi() has been called.
``ti`` is the Quadrants alias from waam_twin.compiler.
"""

from ._common import ALLOY_PLATE, ALLOY_WIRE, bind_velocity_set
from .grid_init import (
    init_grid,
    init_aux_fields,
    clear_forces,
    feed_wire,
    _has_metal_neighbor,
    feed_wire_surface,
    arc_deposition_weight,
    surface_height_at,
    feed_wire_momentum,
    shift_simulation_window_x,
    shift_simulation_window_y,
    shift_simulation_window_z,
    shift_tracers,
    shift_tracers_x,
    mix_alloy_fusion_zone,
    inject_tracers,
    advect_tracers,
)
from .thermal import (
    inject_arc_heat,
    _goldak_pdf_weight,
    inject_goldak_heat,
    update_phase,
    _minmod,
    _masked_at,
    _limited_upwind_grad,
    advect_diffuse_temperature,
    refresh_thermal_properties,
    refresh_thermal_properties_dual,
    advect_diffuse_temperature_variable,
    update_phase_variable_cp,
    update_phase_dual,
    apply_thermal_boundary_losses_variable,
    clamp_enthalpy_floor,
    clamp_enthalpy_floor_scalar,
    apply_evaporative_enthalpy_sink,
    clamp_enthalpy_ceiling_scalar,
    clamp_enthalpy_ceiling_variable_cp,
    clamp_enthalpy_ceiling_dual,
    update_cooling_rate,
    apply_thermal_boundary_losses,
    update_T_max,
    prescribe_gaussian_pulse,
    sync_T_from_H,
    init_stefan_liquid_column,
    clamp_substrate_enthalpy,
    update_time_above_T,
    update_time_above_T_dual,
)
from .vof import (
    _solid_wall_normal,
    _correct_normal_contact_angle,
    _phi_unit_normal_at,
    _brackbill_curvature_at,
    compute_csf_tension,
    compute_csf_wetting,
    apply_contact_angle_phi_bc,
    compute_marangoni_force,
    compute_marangoni_force_variable,
    _vof_face_flux,
    advect_phi,
    reinitialize_phi,
    update_flags_from_phi,
    solidify_cooled_metal,
    solidify_cooled_metal_dual,
    remelt_hot_solid,
    remelt_hot_solid_dual,
    remelt_hot_solid_scalar,
    solidify_trailing_pool,
    solidify_trailing_pool_scalar,
    solidify_trailing_pool_dual,
    add_buoyancy,
    add_hydrostatic_gravity,
)
from .weld import (
    apply_vapor_recoil,
    _wf_pressure_to_Fz_lu,
    _sigma_at,
    apply_vapor_recoil_clausius_clapeyron,
    apply_gas_shear_stress,
    apply_droplet_impact_pressure,
    feed_wire_momentum_impact,
    apply_arc_pressure,
)
from .lorentz import (
    elec_build_sigma,
    elec_clear_source,
    elec_inject_arc_source,
    elec_normalize_source,
    elec_init_ground,
    elec_jacobi_step,
    elec_compute_J,
    elec_l1_diff,
    elec_bin_axial_current,
    elec_prefix_bins,
    elec_B_axisymmetric,
    apply_lorentz_JxB,
)
from .lbm import (
    init_poiseuille_channel,
    set_uniform_Fx,
    collide_srt,
    collide_srt_variable_tau,
    stream,
    clamp_body_force_magnitude,
    clamp_velocity_mach,
    snapshot_forces,
    compute_curvature_field,
    compute_vorticity_magnitude,
    telemetry_pool_reduce,
    count_near_vapor_cap,
    extract_fl_yz_slice,
    sync_phi_liquid_fraction,
)

# Re-export table helper used by thermal kernels and tests.
from ._common import lookup_table_1d, _alloy_pick, _cell_rho

__all__ = [n for n in globals() if not n.startswith("__")]
