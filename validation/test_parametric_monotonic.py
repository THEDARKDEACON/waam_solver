"""
test_parametric_monotonic.py — Monotonic trends: power ↑ → T ↑, dwell ↑ → T ↑.

Uses cell-per-step torch motion so the test completes in CI time while still
coupling travel speed to dwell time at a fixed probe cell.
"""

from __future__ import annotations

import sys

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin


def _probe_after_pass(
    arc_W: float,
    cells_per_step: float,
    n_steps: int = 240,
    *,
    park_at_probe: bool = False,
) -> float:
    twin = WAAMTwin(
        nx=64, ny=32, nz=32, dx=3e-4,
        arc_power_W=arc_W,
        enable_heat_loss=False,
        arc_surface_weighting=True,
        arc_sigma_mm=1.5,
        max_tracers=50,
    )
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    j = g.ny // 2
    k_sub = max(1, twin.nz_solid - 1)
    i_probe = 28

    if park_at_probe:
        # Dwell gate: hold torch on the probe cell.
        x = i_probe * g.dx
        for _ in range(n_steps):
            twin.step(x, cy, is_welding=True)
        return float(g.T.to_numpy()[i_probe, j, k_sub])

    # Power gate: bounce so the probe sees the arc repeatedly.
    x_cells = 8.0
    direction = 1.0
    x_lo, x_hi = 6.0, float(g.nx - 6)
    for _ in range(n_steps):
        x_cells += direction * cells_per_step
        if x_cells >= x_hi:
            x_cells = x_hi
            direction = -1.0
        elif x_cells <= x_lo:
            x_cells = x_lo
            direction = 1.0
        twin.step(x_cells * g.dx, cy, is_welding=True)

    return float(g.T.to_numpy()[i_probe, j, k_sub])


def run() -> float:
    init_taichi(backend="cpu")
    t_low = _probe_after_pass(2000.0, cells_per_step=0.45)
    t_high = _probe_after_pass(3600.0, cells_per_step=0.45)
    # Dwell contrast: parked torch vs fast traverse past the probe.
    t_dwell = _probe_after_pass(2800.0, cells_per_step=0.0, n_steps=160, park_at_probe=True)
    t_fast = _probe_after_pass(2800.0, cells_per_step=1.2, n_steps=160)

    print(
        f"[parametric] probe T — P: {t_low:.0f}K→{t_high:.0f}K  |  "
        f"dwell: park={t_dwell:.0f}K fast={t_fast:.0f}K"
    )
    if t_high <= t_low + 5.0:
        raise AssertionError("Probe T should rise with arc power")
    if t_fast >= t_dwell - 5.0:
        raise AssertionError("Probe T should rise with slower torch traverse (more dwell)")
    return t_high - t_low


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
