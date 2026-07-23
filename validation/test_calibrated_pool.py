"""
test_calibrated_pool.py — Reproduce calibrated full-physics W/D vs macrograph.
"""

from __future__ import annotations

import os
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.calibration import load_calibration, apply_calibration
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(n_steps: int | None = None, threshold_pct: float = 25.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    macro = job.get("reference", {})
    W_macro = float(macro.get("pool_width_mm", 7.0))
    D_macro = float(macro.get("pool_depth_mm", 3.0))
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])

    cal = load_calibration(job.get("calibration"))
    twin = WAAMTwin(
        material=job["material"],
        nx=88, ny=44, nz=44, dx=3e-4,
        arc_power_W=arc_w,
        arc_efficiency=proc.get("arc_efficiency", 0.72),
        T_ambient=float(proc.get("T_ambient_K", 300)),
        heat_source=str(job.get("heat_source", "gaussian2d")),
        max_tracers=100,
    )
    apply_job_to_twin(twin, job)
    apply_calibration(twin, cal)
    twin.travel_speed_m_s = travel
    twin.reset()

    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, _ = measure_pool_mm(twin)
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
