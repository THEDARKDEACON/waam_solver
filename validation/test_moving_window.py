"""
test_moving_window.py — Long path shifts window without NaN.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run(n_steps: int = 1500) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=64, ny=32, nz=28, dx=3e-4,
        arc_power_W=2800.0,
        travel_speed_m_s=0.05,
        enable_moving_window=True,
        max_tracers=50,
    )
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    x_start = 0.010

    for step in range(n_steps):
        x_world = x_start + step * twin.travel_speed_m_s * g.dt
        twin.step(x_world, cy, is_welding=True)
        if step % 400 == 399:
            T = g.T.to_numpy()
            if not np.isfinite(T).all():
                raise AssertionError(f"NaN at step {step + 1}")

    offset = twin._window_offset_x_m
    print(f"[moving_window] offset={offset*1e3:.2f}mm  steps={n_steps}")
    if offset < 1e-6:
        raise AssertionError("Moving window never shifted — increase steps or speed")
    return offset * 1000


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
