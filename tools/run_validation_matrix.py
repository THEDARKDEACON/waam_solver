"""
run_validation_matrix.py — Phase 4 validation matrix with macrograph gating.

Usage:
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.run_validation_matrix
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.run_validation_matrix --quick
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.calibration import load_calibration, apply_calibration
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, plan_path_run_steps, run_bead_travel
from waam_twin.validation.metadata import build_run_metadata

_MATRIX_THRESHOLD_PCT = 15.0

_CASES = [
    {"name": "bead_5mms", "job": "jobs/examples/bead_on_plate.yaml", "travel_mm_s": 5, "preset": None},
    {"name": "bead_8mms", "job": "jobs/examples/bead_on_plate.yaml", "travel_mm_s": 8, "preset": None},
    {"name": "bead_3mms", "job": "jobs/examples/bead_on_plate.yaml", "travel_mm_s": 3, "preset": None},
    {"name": "ss316l_smoke", "job": "jobs/examples/bead_on_plate_ss316l.yaml", "travel_mm_s": 5, "preset": None},
    {"name": "multi_bead", "job": "jobs/examples/multi_bead.yaml", "travel_mm_s": 5, "preset": "minimal", "path": True},
    {"name": "two_layer", "job": "jobs/examples/two_layer.yaml", "travel_mm_s": 5, "preset": "minimal", "path": True},
]


def _run_case(case: dict, n_steps: int) -> dict:
    job = load_job_config(case["job"])
    proc = dict(job.get("process", {}))
    proc["travel_speed_mm_s"] = float(case.get("travel_mm_s", proc.get("travel_speed_mm_s", 5)))
    job = {**job, "process": proc}

    if case.get("preset"):
        os.environ["WAAM_PRESET"] = case["preset"]
        twin = WAAMTwin.from_job(case["job"])
    else:
        twin = WAAMTwin(
            material=job["material"],
            nx=88, ny=44, nz=44, dx=3.0e-4,
            arc_power_W=float(proc["current_A"]) * float(proc["voltage_V"]),
            arc_efficiency=float(proc.get("arc_efficiency", 0.72)),
            T_ambient=float(proc.get("T_ambient_K", 300)),
            heat_source=str(job.get("heat_source", "gaussian2d")),
            max_tracers=150,
        )
        apply_job_to_twin(twin, job)
        cal = load_calibration(job.get("calibration"))
        apply_calibration(twin, cal)

    twin.reset()
    t0 = time.perf_counter()

    if case.get("path"):
        n_steps_run = plan_path_run_steps(twin, job, n_steps=n_steps)
        twin.run_path(case["job"], n_steps=n_steps_run, interpass_steps=40)
    else:
        n_steps_run, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
        run_bead_travel(twin, n_steps_run, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    elapsed = time.perf_counter() - t0
    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    telem = twin.get_telemetry()
    ref = job.get("reference", {})
    W_macro = float(ref.get("pool_width_mm", 7.0))
    D_macro = float(ref.get("pool_depth_mm", 3.0))
    err_macro = pool_error_pct(W_mm or telem["pool_width_mm"], D_mm or telem["pool_depth_mm"], W_macro, D_macro)
    if case.get("path") and W_mm < 0.1:
        W_mm = telem["pool_width_mm"]
        D_mm = telem["pool_depth_mm"]
        err_macro = pool_error_pct(W_mm, D_mm, W_macro, D_macro)

    cal_id = job.get("calibration") or "none"
    meta = build_run_metadata(twin, extra={
        "calibration_id": cal_id,
        "job_path": case["job"],
        "machine_class": os.environ.get("WAAM_MACHINE_CLASS", "unknown"),
    })
    passed = err_macro < _MATRIX_THRESHOLD_PCT
    meta.update({
        "case": case["name"],
        "travel_speed_mm_s": proc["travel_speed_mm_s"],
        "n_steps": n_steps_run,
        "elapsed_s": round(elapsed, 2),
        "pool_width_mm": round(W_mm or telem["pool_width_mm"], 3),
        "pool_depth_mm": round(D_mm or telem["pool_depth_mm"], 3),
        "n_liquid_cells": n_liq,
        "error_pct_macro": round(err_macro, 2),
        "matrix_pass": passed,
        "threshold_pct": _MATRIX_THRESHOLD_PCT,
    })
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation matrix")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="waam_twin/validation/baselines/matrix_latest.json")
    args = parser.parse_args()

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    n_steps = 400 if args.quick else 800
    results = []
    failed = []
    for case in _CASES:
        print(f"[matrix] {case['name']} …")
        row = _run_case(case, n_steps=n_steps)
        results.append(row)
        if not row.get("matrix_pass"):
            failed.append(case["name"])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    v2 = pathlib.Path("waam_twin/validation/baselines/v2.0_matrix.json")
    with open(v2, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[matrix] Wrote {out}  ({len(results)} cases, {len(failed)} macrograph-threshold misses)")
    if failed and os.environ.get("WAAM_MATRIX_STRICT") == "1":
        print(f"[matrix] STRICT failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
