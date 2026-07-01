"""
calibration.py — Scalar overlays for process-specific tuning (v2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationProfile:
    arc_efficiency: float = 1.0
    heat_loss_factor: float = 1.0
    marangoni_scale: float = 1.0
    arc_sigma_scale: float = 1.0


def load_calibration(path: str | None) -> CalibrationProfile:
    if not path:
        return CalibrationProfile()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required for calibration files") from exc
    from pathlib import Path
    from .paths import resolve_project_path

    with open(resolve_project_path(path)) as f:
        data = yaml.safe_load(f) or {}
    return CalibrationProfile(
        arc_efficiency=float(data.get("arc_efficiency", 1.0)),
        heat_loss_factor=float(data.get("heat_loss_factor", 1.0)),
        marangoni_scale=float(data.get("marangoni_scale", 1.0)),
        arc_sigma_scale=float(data.get("arc_sigma_scale", 1.0)),
    )


def apply_calibration(twin, cal: CalibrationProfile) -> None:
    """Apply calibration scalars to a WAAMTwin instance."""
    if cal.arc_efficiency != 1.0:
        twin.eta = cal.arc_efficiency
    if cal.heat_loss_factor != 1.0:
        twin.h_conv *= cal.heat_loss_factor
    if cal.marangoni_scale != 1.0:
        twin.marangoni_scale = cal.marangoni_scale
    if cal.arc_sigma_scale != 1.0:
        twin.sigma_cells *= cal.arc_sigma_scale
