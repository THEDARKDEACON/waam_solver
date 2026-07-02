"""Torch path interpolation for WAAM jobs (robot-agnostic)."""

from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass
from typing import Iterator


@dataclass
class PathSegment:
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    length_m: float
    cumulative_start_m: float = 0.0


def build_segments(waypoints: list[tuple[float, float, float]]) -> list[PathSegment]:
    if len(waypoints) < 2:
        return []
    segs: list[PathSegment] = []
    cum = 0.0
    for i in range(len(waypoints) - 1):
        x0, y0, z0 = waypoints[i]
        x1, y1, z1 = waypoints[i + 1]
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            continue
        segs.append(PathSegment(x0, y0, z0, x1, y1, z1, length, cum))
        cum += length
    return segs


def load_torch_path_csv(path: str | pathlib.Path) -> list[tuple[float, float, float]]:
    """
    Load waypoints from CSV with columns x_mm, y_mm, z_mm (header required).
    """
    path = pathlib.Path(path)
    out: list[tuple[float, float, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((
                float(row.get("x_mm", row.get("x", 0))) / 1000.0,
                float(row.get("y_mm", row.get("y", 0))) / 1000.0,
                float(row.get("z_mm", row.get("z", 0))) / 1000.0,
            ))
    return out


def clamp_torch_to_domain(
    x_m: float,
    y_m: float,
    nx: int,
    ny: int,
    dx: float,
    margin_cells: int = 4,
) -> tuple[float, float]:
    """Keep torch inside the weld domain with a small margin."""
    x_min = margin_cells * dx
    x_max = (nx - margin_cells) * dx
    y_min = margin_cells * dx
    y_max = (ny - margin_cells) * dx
    return (
        max(x_min, min(x_m, x_max)),
        max(y_min, min(y_m, y_max)),
    )


class TorchPathDriver:
    """
    Advance torch along piecewise-linear waypoints at constant travel speed.

    Distance along path is `speed_m_s * sim_time`.
    """

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        travel_speed_m_s: float,
    ):
        self.speed = max(travel_speed_m_s, 1e-9)
        self.segments = build_segments(waypoints)
        self.total_length = sum(s.length_m for s in self.segments)
        if not self.segments and waypoints:
            self._fallback = waypoints[0]
        else:
            self._fallback = (0.0, 0.0, 0.0)

    def distance_at_step(self, step: int, dt: float) -> float:
        return step * dt * self.speed

    def segment_index_at_distance(self, dist_m: float) -> int:
        if not self.segments:
            return 0
        d = dist_m % max(self.total_length, 1e-9)
        for idx, seg in enumerate(self.segments):
            if d <= seg.length_m:
                return idx
            d -= seg.length_m
        return len(self.segments) - 1

    def segment_end(self, seg_idx: int) -> tuple[float, float, float] | None:
        if not self.segments or seg_idx < 0 or seg_idx >= len(self.segments):
            return None
        s = self.segments[seg_idx]
        return s.x1, s.y1, s.z1

    def position_at_time(self, sim_time_s: float) -> tuple[float, float, float]:
        if not self.segments:
            return self._fallback
        dist = (sim_time_s * self.speed) % max(self.total_length, 1e-9)
        for seg in self.segments:
            if dist <= seg.length_m:
                t = dist / seg.length_m
                return (
                    seg.x0 + t * (seg.x1 - seg.x0),
                    seg.y0 + t * (seg.y1 - seg.y0),
                    seg.z0 + t * (seg.z1 - seg.z0),
                )
            dist -= seg.length_m
        last = self.segments[-1]
        return last.x1, last.y1, last.z1

    def positions_for_steps(
        self,
        n_steps: int,
        dt: float,
        start_step: int = 0,
    ) -> Iterator[tuple[int, float, float, float]]:
        for step in range(start_step, start_step + n_steps):
            t = step * dt
            x, y, z = self.position_at_time(t)
            yield step, x, y, z
