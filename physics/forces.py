"""Body force assembly."""

from .. import kernels

clear_forces = kernels.clear_forces
compute_marangoni_force = kernels.compute_marangoni_force
compute_marangoni_force_variable = kernels.compute_marangoni_force_variable
compute_csf_wetting = kernels.compute_csf_wetting
add_buoyancy = kernels.add_buoyancy
add_hydrostatic_gravity = kernels.add_hydrostatic_gravity
apply_arc_pressure = kernels.apply_arc_pressure
apply_vapor_recoil = kernels.apply_vapor_recoil


def compute_csf_tension(
    phi,
    flags,
    Fx,
    Fy,
    Fz,
    gamma_lu,
    FLAG_SOLID,
    FLAG_GAS,
    nx,
    ny,
    nz,
    enable_wetting: bool = False,
    theta_rad: float = 0.0,
) -> None:
    kernels.compute_csf_tension(
        phi, flags, Fx, Fy, Fz, gamma_lu,
        FLAG_SOLID, FLAG_GAS, nx, ny, nz,
        int(enable_wetting), float(theta_rad),
    )


__all__ = [
    "clear_forces",
    "compute_marangoni_force",
    "compute_marangoni_force_variable",
    "compute_csf_tension",
    "compute_csf_wetting",
    "add_buoyancy",
    "add_hydrostatic_gravity",
    "apply_arc_pressure",
    "apply_vapor_recoil",
]
