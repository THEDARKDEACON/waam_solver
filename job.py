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
    path = pathlib.Path(path)
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


def apply_job_to_twin(twin, job: dict[str, Any]) -> None:
    """Apply heat-loss, process, and calibration sections to an existing twin."""
    process = job.get("process", {})
    freq = _wire_droplet_freq_hz(process)
    if freq is not None:
        twin.droplet_freq = freq

    travel_mm_s = process.get("travel_speed_mm_s")
    if travel_mm_s is not None:
        twin.travel_speed_m_s = float(travel_mm_s) / 1000.0

    sim = job.get("simulation", {})
    if "enable_recoil" in sim:
        twin.enable_recoil = bool(sim["enable_recoil"])
    if "enable_csf_tension" in sim:
        twin.enable_csf_tension = bool(sim["enable_csf_tension"])
    if "enable_vof" in sim:
        twin.enable_vof = bool(sim["enable_vof"])
    if sim.get("enable_substrate_growth"):
        twin.enable_substrate_growth = True
    if sim.get("enable_moving_window"):
        twin.enable_moving_window = True

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
        },
        "heat_source": kwargs.pop("heat_source", "gaussian2d"),
        **kwargs,
    }
