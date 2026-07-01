"""JSON sidecar metadata for research VTK bundles."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..twin import WAAMTwin

WAAM_TWIN_VERSION = "2.0.0"


def build_meta_dict(twin: "WAAMTwin", job_path: str | None = None) -> dict[str, Any]:
    g = twin.grid
    mat = twin.mat
    telem = twin.get_telemetry()
    job = getattr(twin, "_job_config", None)

    meta: dict[str, Any] = {
        "waam_twin_version": WAAM_TWIN_VERSION,
        "step": twin._step_n,
        "sim_time_ms": telem["sim_time_ms"],
        "grid": {
            "nx": g.nx,
            "ny": g.ny,
            "nz": g.nz,
            "dx_mm": round(g.dx * 1000.0, 6),
            "dt_us": round(g.dt * 1e6, 4),
        },
        "origin_mm": [
            round(twin._window_offset_x_m * 1000.0, 4),
            0.0,
            0.0,
        ],
        "window_offset_x_mm": telem["window_offset_x_mm"],
        "preset": twin.preset_name,
        "material": {
            "name": mat.name,
            "status": mat.status,
            "T_solidus_K": mat.T_solidus,
            "T_liquidus_K": mat.T_liquidus,
            "rho_kgm3": mat.rho,
        },
        "physics_flags": {
            "enable_vof": twin.enable_vof,
            "enable_csf_tension": twin.enable_csf_tension,
            "enable_recoil": twin.enable_recoil,
            "use_recoil_clausius_clapeyron": twin.use_recoil_clausius_clapeyron,
            "enable_lorentz": twin.enable_lorentz,
            "enable_gas_shear": twin.enable_gas_shear,
            "enable_droplet_impact_pressure": twin.enable_droplet_impact_pressure,
            "enable_substrate_growth": twin.enable_substrate_growth,
            "enable_moving_window": twin.enable_moving_window,
            "enable_wetting": twin.enable_wetting,
            "enable_hydrostatic_gravity": twin.enable_hydrostatic_gravity,
            "enable_bead_freeze": twin.enable_bead_freeze,
            "enable_ctwd": twin.enable_ctwd,
            "contact_angle_deg": twin.contact_angle_deg,
            "enable_heat_loss": twin.enable_heat_loss,
            "C_darcy": twin.C_darcy,
            "heat_source": twin.heat_source_name,
            "welding_current_A": twin.welding_current_A,
        },
        "unit_conversions": {
            "velocity_lu_to_ms": "u_ms = u_lu * dx / dt",
            "force_lu_to_ms2": "a_ms2 = F_lu * dx / dt^2",
            "pressure_pa": "P = rho_lu * cs^2 * rho_phys/dx * (dx/dt)^2",
        },
        "telemetry": telem,
    }

    if job_path:
        meta["job_path"] = job_path
    if job:
        meta["process"] = job.get("process", {})
        meta["reference"] = job.get("reference", {})
    return meta


def write_meta_json(twin: "WAAMTwin", path: str | pathlib.Path, job_path: str | None = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(build_meta_dict(twin, job_path), f, indent=2)
