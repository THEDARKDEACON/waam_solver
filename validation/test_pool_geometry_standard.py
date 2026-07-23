"""
test_pool_geometry_standard.py — Pool W/D at benchmark dx vs macrograph.

Full 266³ standard domain is too slow for CPU CI; uses 88×44×44 @ 0.3 mm.
Set WAAM_STANDARD_VALIDATION=1 to run in CI.
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


def run(n_steps: int | None = None, threshold_pct: float = 20.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    ref = job.get("reference", {})
    W_ref = float(ref.get("pool_width_mm", 7.0))
    D_ref = float(ref.get("pool_depth_mm", 3.0))
    travel = float(job.get("process", {}).get("travel_speed_mm_s", 5.0)) / 1000.0
    proc = job["process"]
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])

    twin = WAAMTwin(
        material=job["material"],
        nx=88, ny=44, nz=44,
        dx=3.0e-4,
        arc_power_W=arc_w,
        arc_efficiency=float(proc.get("arc_efficiency", 0.72)),
        T_ambient=float(proc.get("T_ambient_K", 300)),
        heat_source=str(job.get("heat_source", "gaussian2d")),
        max_tracers=200,
    )
    apply_job_to_twin(twin, job)
    cal = load_calibration(job.get("calibration"))
    apply_calibration(twin, cal)
    twin.travel_speed_m_s = travel
    twin.reset()
    g = twin.grid

    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    if n_liq < 5:
        raise AssertionError("No liquid cells at standard dx")

    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    print(
        f"[pool_geometry_standard] dx={g.dx*1e3:.3f}mm  "
        f"W={W_mm:.2f}mm D={D_mm:.2f}mm  macro_err={err:.1f}%  (threshold {threshold_pct}%)"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Standard-dx pool error {err:.1f}% >= {threshold_pct}%")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
