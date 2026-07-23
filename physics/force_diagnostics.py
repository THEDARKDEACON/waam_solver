"""
Per-force diagnostics for PHYSICS_FORCE_CORRECTNESS_SPEC §8.

Samples max |ΔF| (lattice units) by applying each force in isolation on the
current T/φ/f_l state, then restores Fx,Fy,Fz from the post-step snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import forces, weld_forces

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def force_linf_lu(g) -> float:
    """∞-norm of body-force magnitude on the grid [lu/ts²]."""
    fx = g.Fx.to_numpy()
    fy = g.Fy.to_numpy()
    fz = g.Fz.to_numpy()
    return float(np.sqrt(fx * fx + fy * fy + fz * fz).max())


def _arc_ijk(twin: "WAAMTwin") -> tuple[float, float, float]:
    last = getattr(twin, "_last_arc_ijk", None)
    if last is not None:
        return float(last[0]), float(last[1]), float(last[2])
    g = twin.grid
    return (g.nx - 1) * 0.5, (g.ny - 1) * 0.5, float(twin.nz_solid)


def sample_force_diagnostics(twin: "WAAMTwin") -> dict[str, float]:
    """
    Return max |F| for each catalogue force when applied alone (lu).

    Keys match PHYSICS_FORCE_CORRECTNESS_SPEC §8 force_diagnostics.
    """
    g = twin.grid
    arc_i, arc_j, arc_k = _arc_ijk(twin)
    # Preserve live / snapped forces
    Fx_save = g.Fx.to_numpy().copy()
    Fy_save = g.Fy.to_numpy().copy()
    Fz_save = g.Fz.to_numpy().copy()

    out: dict[str, float] = {
        "f_csf_max": 0.0,
        "f_marangoni_max": 0.0,
        "f_arc_max": 0.0,
        "f_lorentz_max": 0.0,
        "f_buoyancy_max": 0.0,
        "f_gravity_max": 0.0,
        "f_gas_shear_max": 0.0,
        "f_recoil_max": 0.0,
        "f_total_snap_max": 0.0,
    }

    def _measure(apply_fn) -> float:
        forces.clear_forces(g.Fx, g.Fy, g.Fz)
        apply_fn()
        return force_linf_lu(g)

    if twin.enable_csf_tension:
        out["f_csf_max"] = _measure(
            lambda: forces.compute_csf_tension(
                g.phi, g.flags, g.Fx, g.Fy, g.Fz,
                twin.gamma_lu, g.FLAG_SOLID, g.FLAG_GAS,
                g.nx, g.ny, g.nz,
                enable_wetting=twin.enable_wetting,
                theta_rad=twin.theta_rad,
            )
        )

    if twin.use_material_tables:
        out["f_marangoni_max"] = _measure(
            lambda: forces.compute_marangoni_force_variable(
                g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
                g.dgamma_lu_field,
                g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
            )
        )
    else:
        out["f_marangoni_max"] = _measure(
            lambda: forces.compute_marangoni_force(
                g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
                twin.dgamma_dT_lu, g.dx,
                g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
            )
        )

    if twin.enable_gas_shear:
        out["f_gas_shear_max"] = _measure(
            lambda: weld_forces.apply_gas_shear(twin, g, arc_i, arc_j, arc_k)
        )

    p_arc = weld_forces.arc_pressure_peak_pa(twin)
    sig = weld_forces.arc_pressure_sigma_cells(twin)
    out["f_arc_max"] = _measure(
        lambda: forces.apply_arc_pressure(
            g.Fz, g.flags, g.phi,
            arc_i, arc_j, arc_k, sig,
            p_arc, g.dt, g.dx, twin.mat.rho,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
    )

    if twin.enable_recoil:
        out["f_recoil_max"] = _measure(
            lambda: weld_forces.apply_recoil(twin, g, arc_i, arc_j, arc_k)
        )

    if twin.enable_hydrostatic_gravity:
        out["f_gravity_max"] = _measure(
            lambda: forces.add_hydrostatic_gravity(
                g.Fz, g.f_l, g.flags, 1.0, twin.g_lu,
                g.FLAG_SOLID, g.FLAG_GAS,
            )
        )

    out["f_buoyancy_max"] = _measure(
        lambda: forces.add_buoyancy(
            g.T, g.Fz, g.f_l, g.flags,
            twin.g_lu, twin.beta_T, twin.mat.T_liquidus, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
        )
    )

    if twin.enable_lorentz:
        out["f_lorentz_max"] = _measure(
            lambda: weld_forces.solve_lorentz(twin, g, arc_i, arc_j, arc_k)
        )

    # Restore and report snap / live total
    g.Fx.from_numpy(Fx_save)
    g.Fy.from_numpy(Fy_save)
    g.Fz.from_numpy(Fz_save)
    out["f_total_snap_max"] = force_linf_lu(g)

    # Round for JSON stability
    return {k: round(float(v), 6) for k, v in out.items()}


def diagnostics_have_nan(diag: dict[str, Any]) -> bool:
    for v in diag.values():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return True
    return False
