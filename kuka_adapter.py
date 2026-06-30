"""
kuka_adapter.py — Thin bridge from KUKA TCP coordinates to waam_twin job space.

No simulation logic here; use WAAM_JOB env var or pass a job path explicitly.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np


def tcp_mm_to_sim_m(tcp_xyz_mm: list[float] | np.ndarray, origin_mm: tuple[float, float, float] = (0, 0, 0)) -> tuple[float, float, float]:
    """Convert KUKA TCP [mm] to simulation metres relative to weld table origin."""
    arr = np.asarray(tcp_xyz_mm, dtype=np.float64)
    ox, oy, oz = origin_mm
    return (
        (float(arr[0]) - ox) / 1000.0,
        (float(arr[1]) - oy) / 1000.0,
        (float(arr[2]) - oz) / 1000.0,
    )


def default_job_path() -> pathlib.Path:
    env = os.environ.get("WAAM_JOB", "jobs/examples/bead_on_plate.yaml")
    p = pathlib.Path(env)
    if not p.is_absolute():
        from .paths import PROJECT_ROOT as root
        p = root / p
    return p


def create_twin_from_env(**kwargs: Any):
    """Create WAAMTwin from WAAM_JOB + WAAM_PRESET environment variables."""
    from waam_twin.platform import init_taichi
    from waam_twin import WAAMTwin

    init_taichi()
    job_path = default_job_path()
    if job_path.exists():
        return WAAMTwin.from_job(job_path, **kwargs)
    preset = os.environ.get("WAAM_PRESET", "standard")
    return WAAMTwin.from_preset(preset, **kwargs)


def step_from_tcp(twin, tcp_xyz_mm: list[float], is_welding: bool = True, origin_mm=(0, 0, 0)) -> None:
    x, y, z = tcp_mm_to_sim_m(tcp_xyz_mm, origin_mm)
    twin.step(x, y, is_welding=is_welding)
