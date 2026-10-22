"""
Phase 5 — thermal-only steppers.

``thermal_tier_step`` is the arc + conduction step for ``physics_tier: thermal``
jobs without wire feed, run at the explicit conduction limit instead of the
Mach-scaled LBM dt. ``thermal_only_step`` / ``run_interpass`` advance cool-down
without the arc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..materials import MaterialProps
    from ..twin import WAAMTwin

SIGMA_SB = 5.670374419e-8

# Fraction of the explicit FTCS limit dx²/(6 α_max).
THERMAL_DT_SAFETY = 0.8

# Physics thermal_tier_step does not run; any of these needs the coupled step.
_FLOW_FEATURES = (
    "enable_vof", "enable_csf_tension", "enable_wetting", "enable_hydrostatic_gravity",
    "enable_bead_freeze", "enable_substrate_growth", "enable_lorentz", "enable_gas_shear",
    "enable_droplet_impact_pressure", "enable_recoil", "enable_marangoni",
    "enable_buoyancy", "enable_arc_pressure",
)


def _alpha_bound_m2_s(mat: "MaterialProps") -> float:
    """max k / (ρ · min cp) over the constants and every table knot."""
    ks = [float(mat.k)] + [float(v) for _, v in (mat.tables.k or [])]
    cps = [float(mat.cp)] + [float(v) for _, v in (mat.tables.cp or [])]
    return max(ks) / (float(mat.rho) * max(min(cps), 1e-9))


def stable_conduction_dt_s(mats: list["MaterialProps"], dx_m: float) -> float:
    """Explicit 7-point conduction limit dx²/(6 α_max) [s]."""
    alpha = max(_alpha_bound_m2_s(m) for m in mats)
    return dx_m * dx_m / (6.0 * alpha)


def job_thermal_dt_s(job: dict, dx_mm: float) -> float | None:
    """Conduction dt for a thermal-tier job, or None to keep the LBM dt.

    ``simulation.thermal_dt``: ``auto`` (default) → THERMAL_DT_SAFETY of the
    explicit limit; a number → that dt in seconds (must be under the limit);
    ``lbm`` → the Mach-scaled LBM dt and the full coupled step.
    """
    from ..materials import load_material

    sim = job.get("simulation") or {}
    if str(sim.get("physics_tier", "")).strip().lower() != "thermal":
        return None
    mode = sim.get("thermal_dt", "auto")
    is_number = isinstance(mode, (int, float)) and not isinstance(mode, bool)
    if not is_number:
        key = str(mode).strip().lower()
        if key == "lbm":
            return None
        if key != "auto":
            raise ValueError(f"simulation.thermal_dt must be auto, lbm or seconds, got {mode!r}")
    wire_feed = float((job.get("process") or {}).get("wire_feed_m_min", 0.0) or 0.0)
    flow_on = [k for k in _FLOW_FEATURES if sim.get(k)]
    if wire_feed > 0.0 or flow_on:
        if is_number:
            raise ValueError(
                f"simulation.thermal_dt={mode} needs wire_feed_m_min 0 and no flow features "
                f"(wire_feed_m_min={wire_feed}, enabled={flow_on})"
            )
        return None
    mats = [load_material(job.get("material", "ER70S-6"))]
    plate_mat = (job.get("plate") or {}).get("material")
    if plate_mat:
        mats.append(load_material(plate_mat))
    limit = stable_conduction_dt_s(mats, float(dx_mm) * 1e-3)
    if is_number:
        if float(mode) > limit:
            raise ValueError(
                f"simulation.thermal_dt={mode} s exceeds the conduction limit {limit:.3e} s"
            )
        return float(mode)
    return THERMAL_DT_SAFETY * limit


def check_thermal_fast_path(twin: "WAAMTwin") -> None:
    """Refuse a conduction-dt twin that would need the LBM step after all."""
    on = [k for k in _FLOW_FEATURES if getattr(twin, k, False)]
    if on or twin.wire_feed_m_s > 0.0:
        raise ValueError(
            f"thermal tier at the conduction dt cannot run {on or ['wire feed']}; "
            "disable them or set simulation.thermal_dt: lbm"
        )
    mats = [twin.mat] + ([twin.plate_mat] if twin.plate_mat is not None else [])
    g = twin.grid
    limit = stable_conduction_dt_s(mats, g.dx)
    if g.dt > limit * (1.0 + 1e-6):
        raise ValueError(
            f"thermal dt {g.dt:.3e} s exceeds the conduction limit {limit:.3e} s at "
            f"dx={g.dx * 1e3:.3f} mm (auto_grid changed dx?)"
        )


def thermal_tier_step(
    twin: "WAAMTwin",
    torch_x_m: float,
    torch_y_m: float,
    is_welding: bool,
    torch_z_m: float | None = None,
) -> None:
    """coupled_step's arc and thermal operators, in the same order, without LBM/VOF/forces."""
    from ..physics import thermal
    from ..physics.electrical_stickout import update_ctwd
    from .coupled_step import (
        _conduct_and_recover,
        _inject_arc,
        _refresh_thermal,
        _resolve_arc_k,
        _update_time_above,
    )

    g = twin.grid
    arc_i = torch_x_m / g.dx
    arc_j = torch_y_m / g.dx
    arc_k = _resolve_arc_k(twin, g, arc_i, arc_j, torch_z_m)
    twin._last_arc_ijk = (float(arc_i), float(arc_j), float(arc_k))
    twin._enthalpy_cap_energy_J_step = 0.0
    if twin.enable_ctwd:
        update_ctwd(twin, g)

    use_fields = _refresh_thermal(twin, g)
    if is_welding:
        _inject_arc(twin, g, arc_i, arc_j, arc_k)
    _conduct_and_recover(twin, g, use_fields)

    thermal.update_T_max(g.T, g.T_max, g.flags, g.FLAG_GAS)
    thermal.update_cooling_rate(g.T, g.T_prev, g.dT_dt, g.flags, g.dt, g.FLAG_GAS)
    _update_time_above(twin, g)
    twin._step_n += 1
    if twin.probe_recorder is not None:
        twin.probe_recorder.record_step(twin)


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
            g.nx, g.ny, g.nz, g.phi, -1,
        )
        thermal.apply_boundary_losses_variable(
            g.H, g.T, g.flags, g.cp_rho_field,
            twin.T_amb, twin.h_conv, twin.eps_rad,
            en_c, en_r, g.dt, g.dx, SIGMA_SB, T_cap,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz, g.energy_acc_J, g.phi, -1,
        )
    else:
        thermal.advect_diffuse_temperature(
            g.H, g.H_tmp, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.cp_rho, g.nx, g.ny, g.nz, g.phi, -1,
        )
        thermal.apply_boundary_losses(
            g.H, g.T, g.flags,
            twin.T_amb, twin.h_conv, twin.eps_rad,
            en_c, en_r, twin.cp_rho, g.dt, g.dx, SIGMA_SB, T_cap,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz, g.energy_acc_J, g.phi, -1,
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
