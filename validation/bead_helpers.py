"""Shared bead-on-plate run helpers for validation."""

from __future__ import annotations


def run_bead_travel(twin, n_steps: int, x_start_m: float = 0.004) -> None:
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    travel = twin.travel_speed_m_s
    for step in range(n_steps):
        x = x_start_m + step * g.dt * travel
        twin.step(x, cy, is_welding=True)
