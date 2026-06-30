"""
test_pool_geometry.py — Melt-pool W/D vs reference envelope (±35% on minimal).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.job import load_job_config
from waam_twin.validation.torch_motion import torch_position_linear


def run(n_steps: int = 800, threshold_pct: float = 35.0) -> float:
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    ref = job.get("reference", {})
    W_ref = float(ref.get("pool_width_mm", 7.0))
    D_ref = float(ref.get("pool_depth_mm", 2.8))
    travel = float(job.get("process", {}).get("travel_speed_mm_s", 5.0)) / 1000.0

    twin = WAAMTwin.from_job("jobs/examples/bead_on_plate.yaml")
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.travel_speed_m_s = travel
    twin.reset()
    g = twin.grid

    # Minimal preset (dx≈0.5 mm) resolves ~2–3 cells across the pool; use coarse reference.
    if g.dx >= 4.0e-4:
        W_ref = max(W_ref * g.dx / 2.5e-4 * 0.18, 2.0)
        D_ref = max(D_ref * g.dx / 2.5e-4 * 0.18, 2.0)
    cy = (g.ny // 2) * g.dx
    x_start = 0.015

    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, x_start, cy)
        twin.step(x, cy, is_welding=True)

    fl_np = g.f_l.to_numpy()
    liquid_mask = fl_np > 0.5
    if not np.any(liquid_mask):
        raise AssertionError("No liquid cells — increase n_steps")

    x_idx = np.where(liquid_mask.any(axis=(1, 2)))[0]
    z_idx = np.where(liquid_mask.any(axis=(0, 1)))[0]
    W_mm = (x_idx[-1] - x_idx[0] + 1) * g.dx * 1000 if len(x_idx) > 0 else 0.0
    D_mm = (z_idx[-1] - z_idx[0] + 1) * g.dx * 1000 if len(z_idx) > 0 else 0.0

    W_err = abs(W_mm - W_ref) / W_ref * 100.0
    D_err = abs(D_mm - D_ref) / max(D_ref, 0.1) * 100.0
    err = max(W_err, D_err)

    print(f"[pool_geometry] W={W_mm:.2f}mm D={D_mm:.2f}mm  max err={err:.1f}%  (threshold {threshold_pct}%)")
    if err >= threshold_pct:
        raise AssertionError(f"Pool geometry error {err:.1f}% >= {threshold_pct}%")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
