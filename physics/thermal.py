"""Thermal physics module (waam_twin v2)."""

from .. import kernels

advect_diffuse_temperature = kernels.advect_diffuse_temperature
advect_diffuse_variable = kernels.advect_diffuse_temperature_variable
apply_boundary_losses = kernels.apply_thermal_boundary_losses
apply_boundary_losses_variable = kernels.apply_thermal_boundary_losses_variable
refresh_properties = kernels.refresh_thermal_properties
update_T_max = kernels.update_T_max
update_cooling_rate = kernels.update_cooling_rate
update_phase_variable_cp = kernels.update_phase_variable_cp
prescribe_gaussian_pulse = kernels.prescribe_gaussian_pulse
sync_T_from_H = kernels.sync_T_from_H
init_stefan_liquid_column = kernels.init_stefan_liquid_column
clamp_substrate_enthalpy = kernels.clamp_substrate_enthalpy
clamp_enthalpy_floor = kernels.clamp_enthalpy_floor
clamp_enthalpy_floor_scalar = kernels.clamp_enthalpy_floor_scalar
clamp_enthalpy_ceiling_scalar = kernels.clamp_enthalpy_ceiling_scalar
clamp_enthalpy_ceiling_variable_cp = kernels.clamp_enthalpy_ceiling_variable_cp

__all__ = [
    "advect_diffuse_temperature",
    "advect_diffuse_variable",
    "apply_boundary_losses",
    "apply_boundary_losses_variable",
    "refresh_properties",
    "update_T_max",
    "update_cooling_rate",
    "update_phase_variable_cp",
    "prescribe_gaussian_pulse",
    "sync_T_from_H",
    "init_stefan_liquid_column",
    "clamp_substrate_enthalpy",
    "clamp_enthalpy_floor",
    "clamp_enthalpy_floor_scalar",
    "clamp_enthalpy_ceiling_scalar",
    "clamp_enthalpy_ceiling_variable_cp",
]
