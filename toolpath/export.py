"""Export KRL/G-code toolpath segments to waam_twin torch_path CSV."""

from __future__ import annotations

import csv
import pathlib
from typing import Any


def segments_to_waypoints(
    segments: list[dict[str, Any]],
    weld_only: bool = True,
    dedupe_mm: float = 0.05,
) -> list[tuple[float, float, float]]:
    """
    Flatten KRL-style segments [{torch_state, points: [[x,y,z],...]}, ...] to waypoints in metres.

    Points are assumed in mm (KRL convention).
    """
    out: list[tuple[float, float, float]] = []
    last: tuple[float, float, float] | None = None
    for seg in segments:
        if weld_only and str(seg.get("torch_state", "TRUE")).upper() != "TRUE":
            continue
        for pt in seg.get("points") or []:
            if len(pt) < 3:
                continue
            wp = (float(pt[0]) / 1000.0, float(pt[1]) / 1000.0, float(pt[2]) / 1000.0)
            if last is not None:
                d = sum((a - b) ** 2 for a, b in zip(wp, last)) ** 0.5
                if d * 1000.0 < dedupe_mm:
                    continue
            out.append(wp)
            last = wp
    return out


def write_torch_path_csv(
    waypoints_m: list[tuple[float, float, float]],
    path: str | pathlib.Path,
    *,
    include_torch_state: bool = False,
    torch_states: list[str] | None = None,
) -> pathlib.Path:
    """Write x_mm,y_mm,z_mm CSV for job torch_path_csv."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = ["x_mm", "y_mm", "z_mm"]
        if include_torch_state:
            fields.append("torch")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (x, y, z) in enumerate(waypoints_m):
            row = {
                "x_mm": round(x * 1000.0, 4),
                "y_mm": round(y * 1000.0, 4),
                "z_mm": round(z * 1000.0, 4),
            }
            if include_torch_state and torch_states:
                row["torch"] = torch_states[i] if i < len(torch_states) else "TRUE"
            w.writerow(row)
    return out


def segments_to_csv_file(
    segments: list[dict[str, Any]],
    path: str | pathlib.Path,
    weld_only: bool = True,
) -> pathlib.Path:
    wps = segments_to_waypoints(segments, weld_only=weld_only)
    return write_torch_path_csv(wps, path)
