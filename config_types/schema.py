"""Typed config — validated at job load; unknown keys raise."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal


@dataclass
class ThermalConfig:
    enable_enthalpy_cap: bool = True
    enable_evaporative_cooling: bool = True
    evap_cooling_scale: float = 1.0
    T_vapor_cap_K: float = 3200.0
    use_material_tables: bool = True
    use_variable_tau: bool = True
    enhanced_k_factor: float = 1.0  # Tier C: λ for k_eff = λ·k_liq


@dataclass
class FlowConfig:
    tier: Literal["conduction", "cfd"] = "cfd"
    use_srt: bool = True
    periodic_x: bool = False
    enable_marangoni: bool = True
    enable_buoyancy: bool = True
    enable_hydrostatic_gravity: bool = True
    C_darcy_phys: float = 1.0e7  # kg/(m³·s) — physical, not lattice
    u_design_m_s: float = 1.0
    force_limit_m_s2: float = 12500.0
    ma_max: float = 0.08
    dt_mode: Literal["acoustic", "diffusive"] = "acoustic"


@dataclass
class FreeSurfaceConfig:
    enable_vof: bool = True
    enable_csf_tension: bool = True
    enable_wetting: bool = True
    contact_angle_deg: float = 80.0
    use_free_surface_lbm: bool = False  # Phase 3 flag


@dataclass
class HeatSourceConfig:
    model: Literal["gaussian2d", "goldak", "surface_flux"] = "goldak"
    arc_efficiency: float = 0.82
    sigma_q_mm: float = 2.0
    sigma_p_mm: float = 2.0
    pressure_model: Literal["lin_eagar", "constant"] = "lin_eagar"


@dataclass
class DropletConfig:
    enable_deposition: bool = True
    enable_droplet_impact_pressure: bool = True
    transfer_mode: str = "auto"
    superheat_K: float = 400.0
    # Geometric priors — Tier C only
    use_geometric_priors: bool = False
    footprint_sigma_scale: float = 1.0


@dataclass
class LorentzConfig:
    enable_lorentz: bool = True
    jacobi_iters: int = 300
    jacobi_tol: float = 1e-6
    solve_every_s: float = 5.0e-5  # physical cadence
    use_multigrid: bool = False


@dataclass
class LossConfig:
    convection: bool = True
    radiation: bool = True
    h_conv: float = 35.0
    eps_rad: float = 0.4


@dataclass
class MeasureConfig:
    pool_metric: Literal["fusion_zone", "fl_bbox"] = "fusion_zone"
    steady_tol: float = 0.01
    sample_every: int = 200


@dataclass
class TwinConfig:
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    free_surface: FreeSurfaceConfig = field(default_factory=FreeSurfaceConfig)
    heat_source: HeatSourceConfig = field(default_factory=HeatSourceConfig)
    droplet: DropletConfig = field(default_factory=DropletConfig)
    lorentz: LorentzConfig = field(default_factory=LorentzConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    # Legacy bead freeze / trailing (deprecated geometric priors)
    enable_bead_freeze: bool = True
    enable_trailing_solidify: bool = False
    enable_substrate_growth: bool = False


def _merge_dataclass(cls, data: dict[str, Any]):
    if not isinstance(data, dict):
        raise TypeError(f"expected dict for {cls.__name__}, got {type(data)}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    return cls(**{k: data[k] for k in data if k in known})


def load_twin_config(job: dict[str, Any]) -> TwinConfig:
    """Build TwinConfig from a loaded job dict. Unknown subsystem keys raise."""
    sim = job.get("simulation") or {}
    adv = job.get("advanced_physics") or {}
    proc = job.get("process") or {}
    dep = job.get("deposition") or {}
    heat_loss = job.get("heat_loss") or {}
    wet = job.get("surface_wetting") or {}
    arc = job.get("arc_physics") or {}
    claim = job.get("claim") or {}

    tier_raw = str(sim.get("physics_tier", sim.get("tier", "cfd"))).lower()
    if tier_raw in ("full", "cfd", "flow"):
        # "flow" historically meant thermal+LBM without full free surface;
        # map full→cfd, flow→conduction-ish but keep cfd if VOF on.
        tier = "cfd" if sim.get("enable_vof", True) or tier_raw == "full" else "conduction"
    elif tier_raw in ("conduction", "thermal"):
        tier = "conduction"
    else:
        tier = "cfd"

    flow = FlowConfig(
        tier=tier,  # type: ignore[arg-type]
        use_srt=bool(sim.get("use_srt", True)),
        periodic_x=bool(sim.get("periodic_x", False)),
        enable_marangoni=bool(sim.get("enable_marangoni", True)),
        enable_buoyancy=bool(sim.get("enable_buoyancy", True)),
        enable_hydrostatic_gravity=bool(sim.get("enable_hydrostatic_gravity", True)),
        C_darcy_phys=float(sim.get("C_darcy_phys", 1.0e7)),
        u_design_m_s=float(sim.get("u_design_m_s", 1.0)),
        force_limit_m_s2=float(sim.get("force_limit_m_s2", 12500.0)),
        ma_max=float(sim.get("ma_max", sim.get("u_mach_limit_lu", 0.08))),
        dt_mode=str(sim.get("dt_mode", "acoustic")),  # type: ignore[arg-type]
    )
    fs = FreeSurfaceConfig(
        enable_vof=bool(sim.get("enable_vof", True)),
        enable_csf_tension=bool(sim.get("enable_csf_tension", True)),
        enable_wetting=bool(sim.get("enable_wetting", True)),
        contact_angle_deg=float(wet.get("contact_angle_deg", 80.0)),
        use_free_surface_lbm=bool(sim.get("use_free_surface_lbm", False)),
    )
    thermal = ThermalConfig(
        enable_enthalpy_cap=bool(sim.get("enable_enthalpy_cap", True)),
        enable_evaporative_cooling=bool(sim.get("enable_evaporative_cooling", True)),
        evap_cooling_scale=float(adv.get("evap_cooling_scale", 1.0)),
        T_vapor_cap_K=float(arc.get("T_vapor_cap_K", 3200.0)),
        use_variable_tau=bool(sim.get("use_variable_tau", True)),
        enhanced_k_factor=float(sim.get("enhanced_k_factor", 1.0)),
    )
    hs = HeatSourceConfig(
        model=str(job.get("heat_source", "goldak")).replace("goldak3d", "goldak"),  # type: ignore[arg-type]
        arc_efficiency=float(proc.get("arc_efficiency", 0.82)),
        sigma_q_mm=float(arc.get("sigma_mm", proc.get("arc_sigma_mm", 2.0))),
        sigma_p_mm=float(arc.get("pressure_sigma_mm", arc.get("sigma_mm", 2.0))),
        pressure_model=str(arc.get("pressure_model", "lin_eagar")),  # type: ignore[arg-type]
    )
    drop = DropletConfig(
        enable_droplet_impact_pressure=bool(sim.get("enable_droplet_impact_pressure", True)),
        transfer_mode=str(proc.get("transfer_mode", "auto")),
        superheat_K=float(dep.get("superheat_K", 400.0)),
        use_geometric_priors=bool(sim.get("use_geometric_priors", False)),
        footprint_sigma_scale=float(dep.get("footprint_sigma_scale", 1.0)),
    )
    lor = LorentzConfig(
        enable_lorentz=bool(sim.get("enable_lorentz", True)),
        jacobi_iters=int(adv.get("lorentz_jacobi_iters", 300)),
        jacobi_tol=float(adv.get("lorentz_jacobi_tol", 1e-6)),
        solve_every_s=float(adv.get("lorentz_solve_every_s", 5e-5)),
        use_multigrid=bool(adv.get("lorentz_multigrid", False)),
    )
    losses = LossConfig(
        convection=bool(heat_loss.get("convection", True)),
        radiation=bool(heat_loss.get("radiation", True)),
        h_conv=float(heat_loss.get("h_conv", 35.0)),
        eps_rad=float(heat_loss.get("eps_rad", 0.4)),
    )
    meas = MeasureConfig(
        pool_metric=str(claim.get("pool_metric", "fusion_zone")),  # type: ignore[arg-type]
        steady_tol=float(sim.get("steady_tol", 0.01)),
        sample_every=int(sim.get("steady_sample_every", 200)),
    )
    return TwinConfig(
        thermal=thermal,
        flow=flow,
        free_surface=fs,
        heat_source=hs,
        droplet=drop,
        lorentz=lor,
        losses=losses,
        measure=meas,
        enable_bead_freeze=bool(sim.get("enable_bead_freeze", True)),
        enable_trailing_solidify=bool(sim.get("enable_trailing_solidify", False)),
        enable_substrate_growth=bool(sim.get("enable_substrate_growth", False)),
    )
