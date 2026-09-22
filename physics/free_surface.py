"""Free-surface VOF operations."""

from .. import kernels

advect_phi = kernels.advect_phi
reinitialize_phi = kernels.reinitialize_phi
update_flags_from_phi = kernels.update_flags_from_phi
sync_phi_liquid_fraction = kernels.sync_phi_liquid_fraction
surface_height_at = kernels.surface_height_at
apply_contact_angle_phi_bc = kernels.apply_contact_angle_phi_bc
remelt_hot_solid = kernels.remelt_hot_solid
remelt_hot_solid_scalar = kernels.remelt_hot_solid_scalar
remelt_hot_solid_dual = kernels.remelt_hot_solid_dual
shift_simulation_window_x = kernels.shift_simulation_window_x
shift_simulation_window_y = kernels.shift_simulation_window_y
shift_simulation_window_z = kernels.shift_simulation_window_z


def solidify_cooled_metal(
    T,
    H,
    f_l,
    phi,
    flags,
    ux,
    uy,
    uz,
    H_sol: float,
    T_solidus,
    zero_velocity: bool,
    FLAG_SOLID,
    FLAG_FLUID,
    FLAG_GAS,
):
    kernels.solidify_cooled_metal(
        T, H, f_l, phi, flags, ux, uy, uz,
        float(H_sol), T_solidus, 1 if zero_velocity else 0,
        FLAG_SOLID, FLAG_FLUID, FLAG_GAS,
    )


def solidify_cooled_metal_dual(
    T,
    H,
    f_l,
    phi,
    flags,
    ux,
    uy,
    uz,
    alloy_id,
    H_sol_w: float,
    T_solidus_w,
    H_sol_p: float,
    T_solidus_p,
    zero_velocity: bool,
    FLAG_SOLID,
    FLAG_FLUID,
    FLAG_GAS,
):
    kernels.solidify_cooled_metal_dual(
        T, H, f_l, phi, flags, ux, uy, uz, alloy_id,
        float(H_sol_w), T_solidus_w, float(H_sol_p), T_solidus_p,
        1 if zero_velocity else 0,
        FLAG_SOLID, FLAG_FLUID, FLAG_GAS,
    )


__all__ = [
    "advect_phi", "reinitialize_phi", "update_flags_from_phi",
    "sync_phi_liquid_fraction",
    "surface_height_at", "solidify_cooled_metal", "solidify_cooled_metal_dual",
    "apply_contact_angle_phi_bc",
    "remelt_hot_solid", "remelt_hot_solid_scalar", "remelt_hot_solid_dual",
    "shift_simulation_window_x",
    "shift_simulation_window_y",
    "shift_simulation_window_z",
]
