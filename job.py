"""Load portable WAAM job YAML files."""

from __future__ import annotations

import pathlib
from typing import Any


def parse_torch_path(job: dict[str, Any]) -> list[tuple[float, float, float]]:
    """
    Parse optional torch path from job YAML or CSV file reference.

    Format:
      torch_path:
        - {x_mm: 0, y_mm: 10, z_mm: 0}
      torch_path_csv: jobs/paths/bead_line.csv
    """
    csv_ref = job.get("torch_path_csv")
    if csv_ref:
        from .torch_path import load_torch_path_csv
        from .paths import PROJECT_ROOT
        p = pathlib.Path(csv_ref)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return load_torch_path_csv(p)

    raw = job.get("torch_path") or []
    out: list[tuple[float, float, float]] = []
    for pt in raw:
        out.append((
            float(pt.get("x_mm", 0)) / 1000.0,
            float(pt.get("y_mm", 0)) / 1000.0,
            float(pt.get("z_mm", 0)) / 1000.0,
        ))
    return out


def load_job_config(path: str | pathlib.Path) -> dict[str, Any]:
    from .paths import resolve_project_path

    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job file not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install pyyaml") from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _wire_droplet_freq_hz(process: dict[str, Any]) -> float | None:
    wfs = process.get("wire_feed_m_min")
    if wfs is None:
        return None
    wire_m_s = float(wfs) / 60.0
    drop_len_m = float(process.get("droplet_length_mm", 3.0)) / 1000.0
    if drop_len_m <= 0:
        return None
    return wire_m_s / drop_len_m


def apply_physics_tier(twin, tier: str) -> None:
    """
    Apply PHYSICS_FORCE_CORRECTNESS_SPEC §4.3 tier defaults.

    Individual simulation.* flags applied later still override.
    """
    key = str(tier or "flow").strip().lower()
    # Aliases / common typos
    if key in ("base", "default", "standard_physics"):
        from . import logging_util as log
        log.warning(
            f"[job] physics_tier '{tier}' is not defined; using 'flow'. "
            f"Valid: thermal | flow | full"
        )
        key = "flow"
    twin.physics_tier = key
    if key == "thermal":
        twin.enable_vof = False
        twin.enable_csf_tension = False
        twin.enable_wetting = False
        twin.enable_hydrostatic_gravity = False
        twin.enable_bead_freeze = False
        twin.enable_lorentz = False
        twin.enable_gas_shear = False
        twin.enable_droplet_impact_pressure = False
        twin.enable_recoil = False
        twin.arc_pressure_model = "constant"
        return
    if key == "flow":
        twin.enable_vof = True
        twin.grid.ensure_vof_buffers()
        twin.enable_csf_tension = True
        twin.enable_wetting = True
        twin.enable_hydrostatic_gravity = True
        twin.enable_bead_freeze = True
        twin.enable_lorentz = False
        twin.enable_gas_shear = False
        twin.enable_droplet_impact_pressure = True
        twin.enable_recoil = False
        twin.arc_pressure_model = "constant"
        return
    if key != "full":
        from . import logging_util as log
        log.warning(
            f"[job] unknown physics_tier '{tier}'; using 'full'. "
            f"Valid: thermal | flow | full"
        )
    # full
    twin.enable_vof = True
    twin.grid.ensure_vof_buffers()
    twin.enable_csf_tension = True
    twin.enable_wetting = True
    twin.enable_hydrostatic_gravity = True
    twin.enable_bead_freeze = True
    twin.enable_lorentz = True
    twin.grid.ensure_lorentz_fields()
    twin.enable_gas_shear = True
    twin.enable_droplet_impact_pressure = True
    # recoil stays off unless job explicitly enables (conduction-mode WAAM)
    twin.arc_pressure_model = "lin_eagar"
    twin.physics_tier = "full"


def resolve_plate_and_domain(job: dict[str, Any], preset_domain_mm: tuple[float, float, float], preset_dx_mm: float) -> dict[str, Any]:
    """Resolve plate thickness and optional domain/dx overrides from a job dict.

    Job keys (any of these work):
      plate:
        thickness_mm: 10.0
        size_mm: [50, 25]       # optional coupon L×W; default = full domain XY
        origin_mm: [15, 7.5]    # optional lower-left; default = centered
        # domain_mm / dx_mm also accepted here
      simulation:
        plate_thickness_mm: 10.0
        plate_size_mm: [50, 25]
        domain_mm: [80, 40, 30]
        dx_mm: 0.3

    Returns domain_mm, dx_mm, plate_thickness_mm, plate_size_mm, plate_origin_mm.
    """
    sim = job.get("simulation", {}) or {}
    plate = job.get("plate", {}) or {}

    domain = plate.get("domain_mm", sim.get("domain_mm"))
    if domain is not None:
        domain_mm = (float(domain[0]), float(domain[1]), float(domain[2]))
    else:
        domain_mm = preset_domain_mm

    dx = plate.get("dx_mm", sim.get("dx_mm"))
    dx_mm = float(dx) if dx is not None else float(preset_dx_mm)

    thick = plate.get("thickness_mm", sim.get("plate_thickness_mm"))
    if thick is None and "substrate_thickness_mm" in sim:
        thick = sim["substrate_thickness_mm"]
    plate_thickness_mm = float(thick) if thick is not None else None

    # Lateral coupon size (mm). None → plate fills the whole domain XY.
    size = plate.get("size_mm", sim.get("plate_size_mm"))
    plate_size_mm = None
    if size is not None:
        plate_size_mm = (float(size[0]), float(size[1]))
    else:
        lx = plate.get("length_mm", sim.get("plate_length_mm"))
        ly = plate.get("width_mm", sim.get("plate_width_mm"))
        if lx is not None and ly is not None:
            plate_size_mm = (float(lx), float(ly))

    origin = plate.get("origin_mm", sim.get("plate_origin_mm"))
    plate_origin_mm = None
    if origin is not None:
        plate_origin_mm = (float(origin[0]), float(origin[1]))

    if plate_size_mm is not None:
        if plate_size_mm[0] > domain_mm[0] * 1.001 or plate_size_mm[1] > domain_mm[1] * 1.001:
            from . import logging_util as log
            log.warning(
                f"[job] plate size {plate_size_mm} mm exceeds domain XY "
                f"{domain_mm[0]}×{domain_mm[1]} mm — clamping to domain."
            )

    if plate_thickness_mm is not None and plate_thickness_mm >= domain_mm[2] * 0.95:
        from . import logging_util as log
        log.warning(
            f"[job] plate_thickness_mm={plate_thickness_mm} is ≥95% of domain Z="
            f"{domain_mm[2]} mm — raise simulation.domain_mm Z (air gap for the bead)."
        )

    return {
        "domain_mm": domain_mm,
        "dx_mm": dx_mm,
        "plate_thickness_mm": plate_thickness_mm,
        "plate_size_mm": plate_size_mm,
        "plate_origin_mm": plate_origin_mm,
    }


def _apply_heat_source(twin, job: dict[str, Any]) -> None:
    from .physics.arc import create_heat_source, goldak_from_job_mm

    name = str(job.get("heat_source", twin.heat_source_name))
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("goldak", "goldak3d", "doubleellipsoid"):
        twin.arc_source = goldak_from_job_mm(twin, job.get("goldak", {}))
    else:
        twin.arc_source = create_heat_source(name)
    twin.heat_source_name = name


def apply_job_to_twin(twin, job: dict[str, Any]) -> None:
    """Apply heat-loss, process, and calibration sections to an existing twin."""
    process = job.get("process", {})
    freq = _wire_droplet_freq_hz(process)
    if freq is not None:
        twin.droplet_freq = freq

    wfs = process.get("wire_feed_m_min")
    if wfs is not None:
        twin.wire_feed_m_s = float(wfs) / 60.0

    if "wire_diameter_mm" in process:
        twin.wire_diameter_m = float(process["wire_diameter_mm"]) / 1000.0
    if "droplet_freq_hz" in process:
        twin.droplet_freq = float(process["droplet_freq_hz"])
    if "pulse_frequency_hz" in process:
        twin.pulse_frequency_hz = float(process["pulse_frequency_hz"])
    if "droplet_transfer_mode" in process:
        twin.droplet_transfer_mode = str(process["droplet_transfer_mode"])
    if "transfer_mode" in process:
        twin.droplet_transfer_mode = str(process["transfer_mode"])
    if "droplet_size_jitter" in process:
        twin.droplet_size_jitter = float(process["droplet_size_jitter"])
    if "impact_lead_angle_deg" in process:
        twin.impact_lead_angle_deg = float(process["impact_lead_angle_deg"])

    travel_mm_s = process.get("travel_speed_mm_s")
    if travel_mm_s is not None:
        twin.travel_speed_m_s = float(travel_mm_s) / 1000.0

    current_A = process.get("current_A")
    if current_A is not None:
        twin.welding_current_A = float(current_A)

    adv = job.get("advanced_physics", {})
    if "gas_jet_velocity_m_s" in adv:
        twin.gas_jet_velocity_m_s = float(adv["gas_jet_velocity_m_s"])
    if "gas_shear_coeff" in adv:
        twin.gas_shear_coeff = float(adv["gas_shear_coeff"])
    if "sigma_liquid_Sm" in adv:
        twin.sigma_liquid_Sm = float(adv["sigma_liquid_Sm"])
    if "sigma_solid_Sm" in adv:
        twin.sigma_solid_Sm = float(adv["sigma_solid_Sm"])
    if "lorentz_jacobi_iters" in adv:
        twin.lorentz_jacobi_iters = int(adv["lorentz_jacobi_iters"])
    if "T_boiling_K" in adv:
        twin.T_boiling_K = float(adv["T_boiling_K"])
    if "L_vapor_J_kg" in adv:
        twin.L_vapor_J_kg = float(adv["L_vapor_J_kg"])
    if "R_spec_vapor_J_kgK" in adv:
        twin.R_spec_vapor_J_kgK = float(adv["R_spec_vapor_J_kgK"])
    if "recoil_accommodation" in adv:
        twin.recoil_accommodation = float(adv["recoil_accommodation"])

    sim = job.get("simulation", {})
    if "physics_tier" in sim:
        apply_physics_tier(twin, str(sim["physics_tier"]))
    if "strict_mode" in sim:
        twin.strict_mode = bool(sim["strict_mode"])
    import os
    if os.environ.get("WAAM_STRICT", "").strip() in ("1", "true", "True", "yes"):
        twin.strict_mode = True

    if "enable_recoil" in sim:
        twin.enable_recoil = bool(sim["enable_recoil"])
    if "enable_csf_tension" in sim:
        twin.enable_csf_tension = bool(sim["enable_csf_tension"])
    if "enable_lorentz" in sim:
        twin.enable_lorentz = bool(sim["enable_lorentz"])
        if twin.enable_lorentz:
            twin.grid.ensure_lorentz_fields()
    if "enable_gas_shear" in sim:
        twin.enable_gas_shear = bool(sim["enable_gas_shear"])
    if "enable_droplet_impact_pressure" in sim:
        twin.enable_droplet_impact_pressure = bool(sim["enable_droplet_impact_pressure"])
    if "use_recoil_clausius_clapeyron" in sim:
        twin.use_recoil_clausius_clapeyron = bool(sim["use_recoil_clausius_clapeyron"])
    if "enable_vof" in sim:
        twin.enable_vof = bool(sim["enable_vof"])
        if twin.enable_vof:
            twin.grid.ensure_vof_buffers()
    if "enable_enthalpy_cap" in sim:
        twin.enable_enthalpy_cap = bool(sim["enable_enthalpy_cap"])
    if "arc_surface_weighting" in sim:
        twin.arc_surface_weighting = bool(sim["arc_surface_weighting"])
    if "enable_substrate_growth" in sim:
        twin.enable_substrate_growth = bool(sim["enable_substrate_growth"])
    if "enable_moving_window" in sim:
        twin.enable_moving_window = bool(sim["enable_moving_window"])
    if "enable_wetting" in sim:
        twin.enable_wetting = bool(sim["enable_wetting"])
    if "enable_hydrostatic_gravity" in sim:
        twin.enable_hydrostatic_gravity = bool(sim["enable_hydrostatic_gravity"])
    if "enable_bead_freeze" in sim:
        twin.enable_bead_freeze = bool(sim["enable_bead_freeze"])
    if "enable_ctwd" in sim:
        twin.enable_ctwd = bool(sim["enable_ctwd"])
    if sim.get("use_torch_z") or sim.get("enable_torch_z"):
        twin.use_torch_z = True

    # Plate / substrate geometry (decoupled from filling the whole domain).
    plate = job.get("plate", {}) or {}
    thick = plate.get("thickness_mm", sim.get("plate_thickness_mm", sim.get("substrate_thickness_mm")))
    size = plate.get("size_mm", sim.get("plate_size_mm"))
    origin = plate.get("origin_mm", sim.get("plate_origin_mm"))
    plate_size = None
    if size is not None:
        plate_size = (float(size[0]), float(size[1]))
    elif plate.get("length_mm") is not None and plate.get("width_mm") is not None:
        plate_size = (float(plate["length_mm"]), float(plate["width_mm"]))
    plate_origin = None
    if origin is not None:
        plate_origin = (float(origin[0]), float(origin[1]))
    twin.apply_plate_geometry(
        plate_thickness_mm=float(thick) if thick is not None else None,
        plate_size_mm=plate_size,
        plate_origin_mm=plate_origin,
    )

    from .frame import apply_frame_from_job
    apply_frame_from_job(twin, job)

    wet = job.get("surface_wetting", {})
    if "contact_angle_deg" in wet:
        twin.contact_angle_deg = float(wet["contact_angle_deg"])
        twin.theta_rad = __import__("math").radians(twin.contact_angle_deg)

    dep = job.get("deposition", {})
    if "superheat_K" in dep:
        twin.deposition_superheat_K = float(dep["superheat_K"])
    if "footprint_sigma_scale" in dep:
        twin.deposition_footprint_sigma_scale = float(dep["footprint_sigma_scale"])
    if "trailing_solidify_lookback_mm" in dep:
        twin.trailing_solidify_lookback_mm = float(dep["trailing_solidify_lookback_mm"])
    if "trailing_solidify_temp_margin_K" in dep:
        twin.trailing_solidify_temp_margin_K = float(dep["trailing_solidify_temp_margin_K"])

    if "layer_height_mm" in job:
        twin.layer_height_m = float(job["layer_height_mm"]) / 1000.0

    if "ctwd_mm" in process:
        twin.ctwd_m = twin.ctwd_nominal_m = float(process["ctwd_mm"]) / 1000.0
    if "stickout_mm" in process:
        twin.stickout_m = float(process["stickout_mm"]) / 1000.0

    elec = job.get("electrical", {})
    if "rho_e_ohm_m" in elec:
        twin.rho_e_ohm_m = float(elec["rho_e_ohm_m"])
    if "eta_stick" in elec:
        twin.eta_stick = float(elec["eta_stick"])

    arc_phys = job.get("arc_physics", {})
    if "sigma_mm" in arc_phys:
        twin.arc_sigma_m = float(arc_phys["sigma_mm"]) / 1000.0
        twin.sigma_cells = twin.arc_sigma_m / twin.grid.dx
    if "arc_sigma_mm" in process:
        twin.arc_sigma_m = float(process["arc_sigma_mm"]) / 1000.0
        twin.sigma_cells = twin.arc_sigma_m / twin.grid.dx
    if "penetration_mm" in arc_phys:
        twin.arc_penetration_m = float(arc_phys["penetration_mm"]) / 1000.0
    if "T_vapor_cap_K" in arc_phys:
        twin.T_vapor_cap_K = float(arc_phys["T_vapor_cap_K"])
    if "surface_weighting" in arc_phys:
        twin.arc_surface_weighting = bool(arc_phys["surface_weighting"])
    if "pressure_model" in arc_phys:
        twin.arc_pressure_model = str(arc_phys["pressure_model"]).strip().lower()
    if "pressure_pa" in arc_phys:
        twin.arc_pressure = float(arc_phys["pressure_pa"])
    if "pressure_sigma_mm" in arc_phys and arc_phys["pressure_sigma_mm"] is not None:
        twin.pressure_sigma_m = float(arc_phys["pressure_sigma_mm"]) / 1000.0

    if job.get("heat_source"):
        _apply_heat_source(twin, job)

    interpass = job.get("interpass", {})
    if interpass:
        twin._interpass_cooling_steps = int(interpass.get("cooling_steps", 0))
        twin._interpass_travel_mm_s = float(interpass.get("travel_speed_mm_s", 0)) / 1000.0 or None

    heat = job.get("heat_loss", {})
    if heat:
        twin.enable_convection = bool(heat.get("convection", False))
        twin.enable_radiation = bool(heat.get("radiation", False))
        twin.enable_heat_loss = twin.enable_convection or twin.enable_radiation
        if "h_conv" in heat:
            twin.h_conv = float(heat["h_conv"])
        if "eps_rad" in heat:
            twin.eps_rad = float(heat["eps_rad"])

    cal_path = job.get("calibration")
    if cal_path:
        from .calibration import apply_calibration, load_calibration
        apply_calibration(twin, load_calibration(cal_path))


def from_process_sheet(
    current_A: float,
    voltage_V: float,
    wire_feed_m_min: float,
    travel_speed_mm_s: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal job dict from weld process parameters."""
    return {
        "simulation": {"preset": kwargs.pop("preset", "standard"), **kwargs.pop("simulation", {})},
        "material": kwargs.pop("material", "materials/validated/ER70S-6.v1.yaml"),
        "process": {
            "current_A": current_A,
            "voltage_V": voltage_V,
            "wire_feed_m_min": wire_feed_m_min,
            "travel_speed_mm_s": travel_speed_mm_s,
            "arc_efficiency": kwargs.pop("arc_efficiency", 0.72),
            "T_ambient_K": kwargs.pop("T_ambient_K", 300.0),
            "droplet_length_mm": kwargs.pop("droplet_length_mm", 3.0),
            "wire_diameter_mm": kwargs.pop("wire_diameter_mm", 1.2),
        },
        "heat_source": kwargs.pop("heat_source", "gaussian2d"),
        **kwargs,
    }
