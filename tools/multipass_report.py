"""
multipass_report.py — Two-layer remelt / HAZ geometry report.

Runs ``bead_calibrate_twolayer.yaml`` with locked calibrate physics and reports
fusion-zone + HAZ extents. Absolute gates activate when ``reference`` has
measured remelt/HAZ numbers (awaiting_measurement: false).

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.multipass_report
  PYTHONPATH=. WAAM_TWOLAYER_STEPS=6000 python3 -m waam_twin.tools.multipass_report --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_multipass_metrics, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.prediction import CALIBRATE_JOB, assert_physics_lock

TWOLAYER_JOB = "jobs/examples/bead_calibrate_twolayer.yaml"


def run_twolayer_metrics(n_steps: int | None = None) -> dict:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config(TWOLAYER_JOB)
    model = job.get("model_reference") or {}
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_TWOLAYER_STEPS", model.get("n_steps", 12000)))
    interpass = int(model.get("interpass_steps", job.get("interpass", {}).get("cooling_steps", 200)))

    twin = WAAMTwin.from_job(TWOLAYER_JOB)
    twin.enable_vof = True
    twin.enable_substrate_growth = True
    twin.use_torch_z = True
    twin.reset()
    twin.run_path(TWOLAYER_JOB, n_steps=n_steps, interpass_steps=interpass)
    m = measure_multipass_metrics(twin)
    m["n_steps"] = n_steps
    m["interpass_steps"] = interpass
    m["n_liquid"] = int(m.get("n_liquid_cells", 0))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Use WAAM_TWOLAYER_STEPS=5000 if unset")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args(argv)

    if args.quick and "WAAM_TWOLAYER_STEPS" not in os.environ:
        os.environ["WAAM_TWOLAYER_STEPS"] = "5000"

    base = load_job_config(CALIBRATE_JOB)
    job = load_job_config(TWOLAYER_JOB)
    assert_physics_lock(base, job, label="twolayer")

    t0 = time.perf_counter()
    m = run_twolayer_metrics()
    ref = job.get("reference") or {}
    awaiting = bool(ref.get("awaiting_measurement", True))

    print(
        f"\n[multipass] steps={m['n_steps']}  pool W/D={m['pool_width_mm']:.2f}/{m['pool_depth_mm']:.2f}  "
        f"fusion W/D={m['fusion_width_mm']:.2f}/{m['fusion_depth_mm']:.2f}  "
        f"HAZ W/D={m['haz_width_mm']:.2f}/{m['haz_depth_mm']:.2f}  "
        f"T_peak={m['haz_T_peak_K']:.0f}K  h={m['bead_height_mm']:.2f}mm"
    )

    status = "pending"
    if not awaiting and ref.get("remelt_depth_mm") is not None:
        # Compare fusion depth to measured remelt depth.
        D_ref = float(ref["remelt_depth_mm"])
        W_ref = float(ref.get("haz_width_mm", m["haz_width_mm"]))
        err = pool_error_pct(m["haz_width_mm"], m["fusion_depth_mm"], W_ref, D_ref)
        thr = float(ref.get("tolerance_pct", 40.0))
        print(f"  vs experiment remelt/HAZ err={err:.1f}% (tol={thr}%)")
        if err >= thr:
            print("FAIL: multipass geometry vs experiment")
            return 1
        status = "pass"
    else:
        print(
            "  PENDING experiment — paste remelt_depth_mm / haz_width_mm into "
            f"{TWOLAYER_JOB} reference (set awaiting_measurement: false)."
        )
        # Soft credibility: HAZ envelope must exist (fusion depth may be 0 on short runs).
        if m["haz_width_mm"] < 1.0 or m["n_haz_cells"] < 20:
            print("FAIL: multipass HAZ envelope undeveloped")
            return 1
        peak_min = float(ref.get("haz_T_peak_min_K", 800))
        peak_max = float(ref.get("haz_T_peak_max_K", 3200))
        if not (peak_min <= m["haz_T_peak_K"] <= peak_max):
            print(f"FAIL: HAZ peak {m['haz_T_peak_K']:.0f}K outside [{peak_min},{peak_max}]")
            return 1
        if m["n_steps"] >= 10000 and m["fusion_depth_mm"] < 0.4 and m["fusion_width_mm"] < 1.0:
            print("FAIL: multipass fusion zone undeveloped at full step budget")
            return 1
        status = "envelope_ok"

    print(f"Elapsed {time.perf_counter() - t0:.0f}s  status={status}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"status": status, "metrics": m, "reference": ref}, f, indent=2)
        print(f"Wrote {args.json}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
