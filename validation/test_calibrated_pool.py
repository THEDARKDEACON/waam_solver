"""
test_calibrated_pool.py — Reproduce calibrated full-physics W/D vs macrograph.

Uses bead_calibrate.yaml (same gate as pool_geometry) on the job's intended
domain — not an undersized manual grid with auto-expanded step counts.
"""

from __future__ import annotations

import os
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(n_steps: int | None = None, threshold_pct: float = 25.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job_path = "jobs/examples/bead_calibrate.yaml"
    job = load_job_config(job_path)
    macro = job.get("reference", {})
    W_macro = float(macro.get("pool_width_mm", 7.0))
    D_macro = float(macro.get("pool_depth_mm", 3.0))
    travel = float(job.get("process", {}).get("travel_speed_mm_s", 5.0)) / 1000.0
    if n_steps is None:
        env = os.environ.get("WAAM_BEAD_STEPS")
        if env:
            n_steps = int(env)
        else:
            n_steps = int((job.get("model_reference") or {}).get("n_steps", 8000))

    twin = WAAMTwin.from_job(job_path)
    twin.travel_speed_m_s = travel
    twin.reset()

    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    if n_liq < 1:
        raise AssertionError("No liquid cells — increase n_steps")
    err = pool_error_pct(W_mm, D_mm, W_macro, D_macro)
    print(
        f"[calibrated_pool] W={W_mm:.2f} D={D_mm:.2f}  "
        f"macro_err={err:.1f}%  (threshold {threshold_pct}%)"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Calibrated pool error {err:.1f}% >= {threshold_pct}%")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
