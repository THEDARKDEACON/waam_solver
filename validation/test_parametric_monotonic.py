"""
test_parametric_monotonic.py — Monotonic trends: power ↑ → T ↑, dwell ↑ → T ↑.

Uses cell-per-step torch motion so the test completes in CI time while still
coupling travel speed to dwell time at a fixed probe cell.
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def _probe_after_pass(arc_W: float, cells_per_step: float, n_steps: int = 180) -> float:
    twin = WAAMTwin(
        nx=64, ny=32, nz=32, dx=3e-4,
        arc_power_W=arc_W,
        enable_heat_loss=False,
        max_tracers=50,
    )
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    j = g.ny // 2
    k_sub = max(1, twin.nz_solid - 1)
    i_probe = 40

    x_cells = 10.0
    for _ in range(n_steps):
        x_cells += cells_per_step
        twin.step(x_cells * g.dx, cy, is_welding=True)

    return float(g.T.to_numpy()[i_probe, j, k_sub])


def run() -> float:
    init_taichi(backend="cpu")
    t_low = _probe_after_pass(2400.0, cells_per_step=0.5)
    t_high = _probe_after_pass(3200.0, cells_per_step=0.5)
    t_fast = _probe_after_pass(2800.0, cells_per_step=1.2)
    t_slow = _probe_after_pass(2800.0, cells_per_step=0.4)

    print(
        f"[parametric] probe T — P: {t_low:.0f}K→{t_high:.0f}K  |  "
        f"dwell: slow={t_slow:.0f}K fast={t_fast:.0f}K"
    )
    if t_high <= t_low + 5.0:
        raise AssertionError("Probe T should rise with arc power")
    if t_fast >= t_slow - 5.0:
        raise AssertionError("Probe T should rise with slower torch traverse (more dwell)")
    return t_high - t_low


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
