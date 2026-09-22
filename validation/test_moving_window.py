"""
test_moving_window.py — Long path shifts window without NaN.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
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
    print(f"[moving_window] offset_x={offset*1e3:.2f}mm  steps={n_steps}")
    if offset < 1e-6:
        raise AssertionError("Moving window never shifted — increase steps or speed")
    if abs(twin._window_offset_y_m) > 1e-9:
        raise AssertionError("centered +X bead must not slide Y")
    _run_yz_direct()
    return offset * 1000


def _run_yz_direct() -> None:
    """Edge torch in Y/Z slides those axes without NaN."""
    twin = WAAMTwin(
        nx=32, ny=64, nz=28, dx=3e-4,
        arc_power_W=2800.0,
        enable_moving_window=True,
        max_tracers=20,
    )
    twin.reset()
    g = twin.grid
    cx_safe = 0.3 * g.nx * g.dx
    cy = (g.ny // 2) * g.dx
    y_hi = (g.ny - 2) * g.dx
    twin._maybe_shift_window(cx_safe, y_hi)
    if twin._window_offset_y_m < 1e-6:
        raise AssertionError("+Y window did not shift at the +Y face")
    T = g.T.to_numpy()
    if not np.isfinite(T).all():
        raise AssertionError("NaN after +Y window shift")

    twin.use_torch_z = True
    z_hi = (g.nz - 1) * g.dx
    z_before = twin.nz_solid
    twin._maybe_shift_window(cx_safe, cy, z_hi)
    if twin._window_offset_z_m < 1e-6:
        raise AssertionError("+Z window did not shift at the +Z face")
    if twin.nz_solid >= z_before:
        raise AssertionError("+Z shift should lower nz_solid")
    print(
        f"[moving_window] offset_y={twin._window_offset_y_m*1e3:.2f}mm  "
        f"offset_z={twin._window_offset_z_m*1e3:.2f}mm  nz_solid={twin.nz_solid}"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
