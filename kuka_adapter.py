"""
kuka_adapter.py — Thin bridge from KUKA TCP coordinates to waam_twin job space.

No simulation logic here; use WAAM_JOB / WAAM_FRAME environment variables or job YAML.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np

from .frame import WeldFrame, apply_frame_to_twin, load_weld_frame


def get_weld_frame(twin=None) -> WeldFrame:
    if twin is not None and getattr(twin, "weld_frame", None) is not None:
        return twin.weld_frame
    return load_weld_frame()


def tcp_mm_to_sim_m(
    tcp_xyz_mm: list[float] | np.ndarray,
    frame: WeldFrame | None = None,
) -> tuple[float, float, float]:
    """Convert KUKA TCP [mm] to world metres using the weld-table origin.

    NOTE: sim_origin_offset_m is intentionally NOT applied here —
    ``WAAMTwin.step()`` subtracts it (single authority). Applying it in both
    places shifted the torch by twice the configured offset.
    """
    frame = frame or load_weld_frame()
    arr = np.asarray(tcp_xyz_mm, dtype=np.float64)
    ox, oy, oz = frame.origin_mm
    return (
        (float(arr[0]) - ox) / 1000.0,
        (float(arr[1]) - oy) / 1000.0,
        (float(arr[2]) - oz) / 1000.0,
    )


def default_job_path() -> pathlib.Path:
    from .paths import resolve_project_path

    env = os.environ.get("WAAM_JOB", "jobs/examples/bead_on_plate.yaml")
    return resolve_project_path(env)


def create_twin_from_env(**kwargs: Any):
    """Create WAAMTwin from WAAM_JOB + WAAM_PRESET + WAAM_FRAME environment variables."""
    from waam_twin.platform import init_taichi
    from waam_twin import WAAMTwin

    init_taichi()
    job_path = default_job_path()
    if job_path.exists():
        twin = WAAMTwin.from_job(job_path, **kwargs)
    else:
        preset = os.environ.get("WAAM_PRESET", "standard")
        twin = WAAMTwin.from_preset(preset, **kwargs)
        apply_frame_to_twin(twin, load_weld_frame())
    return twin


def step_from_tcp(
    twin,
    tcp_xyz_mm: list[float] | np.ndarray,
    is_welding: bool = True,
    frame: WeldFrame | None = None,
) -> None:
    """Advance one timestep from KUKA $POS_ACT-style TCP (mm)."""
    frame = frame or get_weld_frame(twin)
    x, y, z = tcp_mm_to_sim_m(tcp_xyz_mm, frame)
    twin.step(x, y, is_welding=is_welding, torch_z_m=z if getattr(twin, "use_torch_z", False) else None)
