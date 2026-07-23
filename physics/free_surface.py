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
shift_simulation_window_x = kernels.shift_simulation_window_x


def solidify_cooled_metal(
    T,
    f_l,
    phi,
    flags,
    ux,
    uy,
    uz,
    T_solidus,
    zero_velocity: bool,
    FLAG_SOLID,
    FLAG_FLUID,
    FLAG_GAS,
):
    kernels.solidify_cooled_metal(
        T, f_l, phi, flags, ux, uy, uz,
        T_solidus, 1 if zero_velocity else 0,
        FLAG_SOLID, FLAG_FLUID, FLAG_GAS,
    )


__all__ = [
    "advect_phi", "reinitialize_phi", "update_flags_from_phi",
    "sync_phi_liquid_fraction",
    "surface_height_at", "solidify_cooled_metal", "apply_contact_angle_phi_bc",
    "remelt_hot_solid", "remelt_hot_solid_scalar", "shift_simulation_window_x",
]
