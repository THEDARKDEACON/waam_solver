"""Weld-table / robot coordinate frame for KUKA ↔ simulation mapping."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class WeldFrame:
    name: str = "default"
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    substrate_z_mm: float = 0.0
    sim_origin_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def substrate_z_m(self) -> float:
        return self.substrate_z_mm / 1000.0


def load_weld_frame(path: str | None = None) -> WeldFrame:
    """Load frame YAML or fall back to WAAM_FRAME env / default weld_table."""
    from .paths import resolve_project_path

    ref = path or os.environ.get("WAAM_FRAME", "jobs/frames/weld_table.yaml")
    p = resolve_project_path(ref)
    if not p.exists():
        return WeldFrame()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required") from exc
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    origin = data.get("origin_mm", {}) or {}
    off = data.get("sim_origin_offset_m", {}) or {}
    return WeldFrame(
        name=str(data.get("name", p.stem)),
        origin_mm=(
            float(origin.get("x", 0.0)),
            float(origin.get("y", 0.0)),
            float(origin.get("z", 0.0)),
        ),
        substrate_z_mm=float(data.get("substrate_z_mm", 0.0)),
        sim_origin_offset_m=(
            float(off.get("x", 0.0)),
            float(off.get("y", 0.0)),
            float(off.get("z", 0.0)),
        ),
    )


def apply_frame_to_twin(twin, frame: WeldFrame) -> None:
    twin.weld_frame = frame
    twin.frame_origin_mm = frame.origin_mm
    twin.substrate_z_m = frame.substrate_z_m
    ox, oy, _ = frame.sim_origin_offset_m
    twin._sim_origin_offset_x_m = ox
    twin._sim_origin_offset_y_m = oy


def apply_frame_from_job(twin, job: dict[str, Any]) -> None:
    ref = job.get("frame")
    if ref:
        apply_frame_to_twin(twin, load_weld_frame(str(ref)))
