"""
Phase 5 — thermal-only stepper for interpass / cooldown.

Skips LBM collide/stream, VOF, and weld forces. Advances advection–diffusion
(or pure diffusion when u≈0) with radiation/convection losses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin

SIGMA_SB = 5.670374419e-8


def thermal_only_step(twin: "WAAMTwin") -> None:
    from ..physics import thermal
    from ..solvers.coupled_step import _refresh_thermal, _update_phase

    g = twin.grid
    use_fields = _refresh_thermal(twin, g)
    g.H_tmp.copy_from(g.H)
    T_cap = float(getattr(twin, "T_vapor_cap", 3200.0))
    en_c = 1 if getattr(twin, "enable_convection", True) else 0
    en_r = 1 if getattr(twin, "enable_radiation", True) else 0
    if use_fields:
        thermal.advect_diffuse_variable(
            g.H, g.H_tmp, g.T, g.ux, g.uy, g.uz, g.flags,
            g.alpha_lu_field, g.cp_rho_field, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        thermal.apply_boundary_losses_variable(
            g.H, g.T, g.flags, g.cp_rho_field,
            twin.T_amb, twin.h_conv, twin.eps_rad,
            en_c, en_r, g.dt, g.dx, SIGMA_SB, T_cap,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    else:
        thermal.advect_diffuse_temperature(
            g.H, g.H_tmp, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz,
        )
        thermal.apply_boundary_losses(
            g.H, g.T, g.flags,
            twin.T_amb, twin.h_conv, twin.eps_rad,
            en_c, en_r, twin.cp_rho, g.dt, g.dx, SIGMA_SB, T_cap,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    _update_phase(twin, g)
    thermal.update_T_max(g.T, g.T_max, g.flags, g.FLAG_GAS)
    twin._step_n += 1


def run_interpass(twin: "WAAMTwin", duration_s: float) -> int:
    """Advance thermal-only for ``duration_s``. Returns steps taken."""
    dt = float(twin.grid.dt)
    n = max(1, int(round(duration_s / dt)))
    for _ in range(n):
        thermal_only_step(twin)
    return n
