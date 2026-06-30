"""Validation run metadata (Phase 4 evidence trail)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_run_metadata(twin, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    g = twin.grid
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "backend": os.environ.get("WAAM_BACKEND", "auto"),
        "preset": twin.preset_name or os.environ.get("WAAM_PRESET", ""),
        "material_name": twin.mat.name,
        "material_status": twin.mat.status,
        "grid_nx": g.nx,
        "grid_ny": g.ny,
        "grid_nz": g.nz,
        "dx_mm": round(g.dx * 1000, 4),
        "dt_us": round(g.dt * 1e6, 3),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(path: str | Path, twin, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_run_metadata(twin, extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
