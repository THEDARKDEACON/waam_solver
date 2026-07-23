"""Shared bead-on-plate run helpers for validation."""

from __future__ import annotations

import math


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def steps_for_travel(distance_m: float, travel_speed_m_s: float, dt_s: float) -> int:
    """Steps required to traverse a physical distance at the current speed."""
    speed = max(abs(travel_speed_m_s), 1e-12)
    return max(1, int(math.ceil(max(distance_m, 0.0) / (speed * dt_s))))


def plan_linear_bead_run(
    twin,
    job: dict,
    n_steps: int | None = None,
    margin_cells: int = 4,
    default_distance_m: float = 0.02,
) -> tuple[int, float, float, float]:
    """Plan a representative straight-bead validation run.

    Uses the first path segment when available, then clips the travel distance
    to what fits in the current grid with a small margin. Returns
    ``(n_steps, x_start_m, y_m, direction_x)``.

    Honours ``WAAM_BEAD_STEPS`` / ``WAAM_MAX_BEAD_STEPS`` so large job domains
    (long paths, fine dx) do not silently schedule 1e5+ steps.
    """
    import os

    from waam_twin.job import parse_torch_path

    g = twin.grid
    dx = g.dx
    x_min = margin_cells * dx
    x_max = (g.nx - margin_cells) * dx
    y_min = margin_cells * dx
    y_max = (g.ny - margin_cells) * dx

    waypoints = parse_torch_path(job)
    if len(waypoints) >= 2:
        (x0, y0, _z0), (x1, y1, _z1) = waypoints[0], waypoints[1]
        dir_x = 1.0 if x1 >= x0 else -1.0
        x_start = _clamp(x0, x_min, x_max)
        y_m = _clamp(y0, y_min, y_max)
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    else:
        dir_x = 1.0
        x_start = max(0.004, x_min)
        y_m = (g.ny // 2) * dx
        seg_len = default_distance_m

    usable = (x_max - x_start) if dir_x >= 0.0 else (x_start - x_min)
    distance_m = max(0.0, min(seg_len, usable))
    if distance_m <= 0.0:
        distance_m = min(default_distance_m, max(x_max - x_min, dx))
    derived_steps = steps_for_travel(distance_m, twin.travel_speed_m_s, g.dt)

    env_steps = os.environ.get("WAAM_BEAD_STEPS")
    if n_steps is not None:
        out_steps = int(n_steps)
    elif env_steps:
        out_steps = int(env_steps)
    else:
        out_steps = derived_steps

    max_steps = os.environ.get("WAAM_MAX_BEAD_STEPS")
    if max_steps:
        out_steps = min(out_steps, int(max_steps))

    return out_steps, x_start, y_m, dir_x


def plan_path_run_steps(
    twin,
    job: dict,
    n_steps: int | None = None,
) -> int:
    """Use the actual job path length to size a validation run."""
    from waam_twin.job import parse_torch_path
    from waam_twin.torch_path import build_segments

    if n_steps is not None:
        return int(n_steps)
    segs = build_segments(parse_torch_path(job))
    total_len = sum(s.length_m for s in segs)
    if total_len <= 0.0:
        total_len = 0.02
    return steps_for_travel(total_len, twin.travel_speed_m_s, twin.grid.dt)


def run_bead_travel(
    twin,
    n_steps: int,
    x_start_m: float = 0.004,
    y_m: float | None = None,
    direction_x: float = 1.0,
) -> None:
    g = twin.grid
    cy = (g.ny // 2) * g.dx if y_m is None else y_m
    travel = twin.travel_speed_m_s
    for step in range(n_steps):
        x = x_start_m + direction_x * step * g.dt * travel
        twin.step(x, cy, is_welding=True)
