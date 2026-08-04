"""
recoil_accommodation_sweep.py — Evidence sweep of C_acc vs W/D and crown.

Keeps Goldak/η/T_boil/evap fixed on the calibrate job; only varies
``recoil_accommodation``. Use before raising C_acc above the locked 0.25.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.recoil_accommodation_sweep
  PYTHONPATH=. WAAM_BEAD_STEPS=2000 python3 -m waam_twin.tools.recoil_accommodation_sweep --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics, measure_pool_mm
from waam_twin.job import load_job_config
from waam_twin.platform import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel
from waam_twin.validation.prediction import CALIBRATE_JOB


def _run_one(c_acc: float, n_steps: int) -> dict:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config(CALIBRATE_JOB)
    twin = WAAMTwin.from_job(CALIBRATE_JOB)
    twin.recoil_accommodation = float(c_acc)
    twin.reset()
    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)
    W, D, n_liq = measure_pool_mm(twin)
    bead = measure_bead_metrics(twin)
    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    return {
        "recoil_accommodation": float(c_acc),
        "n_steps": n_steps,
        "pool_width_mm": float(W),
        "pool_depth_mm": float(D),
        "n_liquid": int(n_liq),
        "bead_height_mm": float(bead.get("bead_height_mm", telem.get("bead_height_mm", 0))),
        "peak_temp_C": float(telem.get("peak_temp_C", 0)),
        "n_cap": int(telem.get("n_cells_at_vapor_cap", 0)),
        "f_recoil_max": float(fd.get("f_recoil_max", 0)),
        "vapor_cap_saturated": bool(telem.get("vapor_cap_saturated")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--values",
        type=str,
        default="0.10,0.25,0.40,0.54",
        help="Comma-separated C_acc values",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args(argv)

    if args.quick and "WAAM_BEAD_STEPS" not in os.environ:
        os.environ["WAAM_BEAD_STEPS"] = "4000"
        print("NOTE: --quick uses 4000 steps; prefer full 8000 for lock decisions.")

    job = load_job_config(CALIBRATE_JOB)
    model = job.get("model_reference") or {}
    n_steps = int(os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000)))
    values = [float(x.strip()) for x in args.values.split(",") if x.strip()]

    locked = float((job.get("advanced_physics") or {}).get("recoil_accommodation", 0.25))
    print(
        f"Recoil C_acc sweep on {CALIBRATE_JOB}\n"
        f"  locked C_acc={locked}  steps={n_steps}  "
        f"Goldak/η/T_boil/evap frozen\n"
    )
    print(
        f"{'C_acc':>6}  {'W_mm':>6}  {'D_mm':>6}  {'h_mm':>6}  "
        f"{'T_C':>6}  {'n_cap':>5}  {'f_recoil':>10}  sat"
    )

    rows: list[dict] = []
    t0 = time.perf_counter()
    for c in values:
        row = _run_one(c, n_steps)
        rows.append(row)
        print(
            f"{row['recoil_accommodation']:6.2f}  "
            f"{row['pool_width_mm']:6.2f}  {row['pool_depth_mm']:6.2f}  "
            f"{row['bead_height_mm']:6.2f}  {row['peak_temp_C']:6.0f}  "
            f"{row['n_cap']:5d}  {row['f_recoil_max']:10.2e}  "
            f"{'YES' if row['vapor_cap_saturated'] else 'no'}"
        )

    print(f"\nElapsed {time.perf_counter() - t0:.0f}s")
    print(
        "Guidance: raise C_acc only if W/D vs macrograph improves AND n_cap stays ~0.\n"
        f"Current lock remains C_acc={locked} until evidence says otherwise."
    )

    # Soft evidence gate for CI smoke: locked value must appear and stay unsaturated.
    locked_row = next((r for r in rows if abs(r["recoil_accommodation"] - locked) < 1e-9), None)
    if locked_row is None:
        print("WARN: locked C_acc not in sweep list")
    elif locked_row["vapor_cap_saturated"] or locked_row["n_cap"] > 0:
        print("FAIL: locked C_acc saturates vapor cap — do not raise further")
        return 1
    elif locked_row["f_recoil_max"] <= 0.0:
        print("FAIL: locked C_acc produces no recoil force")
        return 1

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"locked": locked, "rows": rows}, f, indent=2)
        print(f"Wrote {args.json}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
