"""Torch motion helpers for process validation."""

from __future__ import annotations


def advance_torch(
    x_m: float,
    travel_speed_m_s: float,
    dt: float,
    x_min: float,
    x_max: float,
) -> float:
    """Advance torch along +x, bouncing at domain edges."""
    x_new = x_m + travel_speed_m_s * dt
    if x_new > x_max or x_new < x_min:
        return x_m  # hold at boundary for soak tests
    return x_new


def torch_position_linear(
    step: int,
    dt: float,
    travel_speed_m_s: float,
    x_start_m: float,
    y_m: float,
) -> tuple[float, float]:
    return x_start_m + step * travel_speed_m_s * dt, y_m
