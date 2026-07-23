"""
test_pool_geometry.py — Melt-pool W/D vs macrograph on benchmark grid.
"""

from __future__ import annotations

import os
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(n_steps: int | None = None, threshold_pct: float = 30.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config("jobs/examples/bead_calibrate.yaml")
    ref = job.get("reference", {})
    W_ref = float(ref.get("pool_width_mm", 7.0))
    D_ref = float(ref.get("pool_depth_mm", 3.0))
    travel = float(job.get("process", {}).get("travel_speed_mm_s", 5.0)) / 1000.0
    if n_steps is None:
        env = os.environ.get("WAAM_BEAD_STEPS")
        if env:
            n_steps = int(env)
        else:
            n_steps = int((job.get("model_reference") or {}).get("n_steps", 8000))

    twin = WAAMTwin.from_job("jobs/examples/bead_calibrate.yaml")
    twin.travel_speed_m_s = travel
    twin.reset()
    g = twin.grid

    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    if n_liq < 1:
        raise AssertionError("No liquid cells — increase n_steps")

    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    telem = twin.get_telemetry()

    print(
        f"[pool_geometry] dx={g.dx*1e3:.3f}mm  steps={n_steps}  "
        f"W={W_mm:.2f}mm D={D_mm:.2f}mm  "
        f"macro_err={err:.1f}%  (threshold {threshold_pct}%)  "
        f"T_peak={telem['peak_temp_C']:.0f}C  bal={telem['mass_balance_ratio']:.2f}"
    )
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
