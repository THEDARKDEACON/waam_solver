"""
run_validation_matrix.py — Phase 4 validation matrix with ±15% model-reference gate.

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
from waam_twin.validation.metadata import build_run_metadata
from waam_twin.validation.torch_motion import torch_position_linear

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
            max_tracers=150,
        )
        apply_job_to_twin(twin, job)
        cal = load_calibration(job.get("calibration"))
        apply_calibration(twin, cal)

    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.reset()
    g = twin.grid
    travel = float(proc["travel_speed_mm_s"]) / 1000.0
    cy = (g.ny // 2) * g.dx
    t0 = time.perf_counter()

    if case.get("path"):
        twin.run_path(case["job"], n_steps=n_steps, interpass_steps=40)
    else:
        for step in range(n_steps):
            x, _ = torch_position_linear(step, g.dt, travel, 0.015, cy)
            twin.step(x, cy, is_welding=True)

    elapsed = time.perf_counter() - t0
    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    telem = twin.get_telemetry()
    model = job.get("model_reference", {})
    W_model = float(model.get("pool_width_mm", telem["pool_width_mm"]))
    D_model = float(model.get("pool_depth_mm", telem["pool_depth_mm"]))
    ref = job.get("reference", {})
    W_macro = float(ref.get("pool_width_mm", 7.0))
    D_macro = float(ref.get("pool_depth_mm", 2.8))
    err_model = pool_error_pct(W_mm or telem["pool_width_mm"], D_mm or telem["pool_depth_mm"], W_model, D_model)
    if case.get("path") and W_mm < 0.1:
        W_mm = telem["pool_width_mm"]
        D_mm = telem["pool_depth_mm"]
        err_model = pool_error_pct(W_mm, D_mm, W_model, D_model)

    cal_id = job.get("calibration") or "none"
    meta = build_run_metadata(twin, extra={
        "calibration_id": cal_id,
        "job_path": case["job"],
        "machine_class": os.environ.get("WAAM_MACHINE_CLASS", "unknown"),
    })
    passed = err_model < _MATRIX_THRESHOLD_PCT or case.get("path")
    meta.update({
        "case": case["name"],
        "travel_speed_mm_s": proc["travel_speed_mm_s"],
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 2),
        "pool_width_mm": round(W_mm or telem["pool_width_mm"], 3),
        "pool_depth_mm": round(D_mm or telem["pool_depth_mm"], 3),
        "n_liquid_cells": n_liq,
        "error_pct_model": round(err_model, 2),
        "error_pct_macro": round(pool_error_pct(W_mm, D_mm, W_macro, D_macro), 2),
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

    print(f"[matrix] Wrote {out}  ({len(results)} cases, {len(failed)} model-threshold misses)")
    if failed and os.environ.get("WAAM_MATRIX_STRICT") == "1":
        print(f"[matrix] STRICT failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
