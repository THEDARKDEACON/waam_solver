"""
prediction_report.py — Fitted vs predicted credibility report.

Runs the locked ``bead_calibrate`` baseline and held-out process variants
with Goldak/η/recoil frozen. Prints W/D, crown, T, trend gates, and
macrograph absolute error when ``reference`` is filled.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.prediction_report
  PYTHONPATH=. WAAM_BEAD_STEPS=3000 python3 -m waam_twin.tools.prediction_report --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from waam_twin.job import load_job_config
from waam_twin.validation.prediction import (
    CALIBRATE_JOB,
    HELDOUT_BRUNO_JOB,
    HELDOUT_FAST_JOB,
    HELDOUT_HOT_JOB,
    HELDOUT_MACRO2_JOB,
    assert_macrograph_prediction,
    assert_physics_lock,
    assert_trend,
    run_job_metrics,
)


def _fmt(m: dict) -> str:
    err = m.get("macro_err_pct")
    if err is not None:
        err_s = f"  macro_err={err:.1f}%"
    elif m.get("awaiting_measurement"):
        err_s = "  (awaiting macrograph measurement)"
    else:
        err_s = "  (prediction-only)"
    return (
        f"I={m['current_A']:.0f}A  V={m['voltage_V']:.0f}V  v={m['travel_mm_s']:.1f}mm/s  "
        f"Q={m['Q_w_W']:.0f}W  η={m['eta']:.2f}  C_acc={m['recoil_accommodation']:.2f}  "
        f"evap_scale={m.get('evap_cooling_scale', 25):.1f}\n"
        f"  W={m['pool_width_mm']:.2f}  D={m['pool_depth_mm']:.2f} mm  "
        f"h={m['bead_height_mm']:.2f}  T={m['peak_temp_C']:.0f}C  "
        f"n_cap={m['n_cap']}  f_recoil={m['f_recoil_max']:.2e}"
        f"{err_s}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Metrics-only smoke (4000 steps); skips trend gates (pool may be undeveloped)",
    )
    ap.add_argument("--json", type=str, default="", help="Write results JSON to path")
    ap.add_argument("--skip-trends", action="store_true", help="Do not fail on trend gates")
    ap.add_argument(
        "--with-bruno",
        action="store_true",
        help="Also run Bruno GMAW surface-bead held-out (PIONEER Bruno_Dataset)",
    )
    args = ap.parse_args(argv)

    if args.quick and "WAAM_BEAD_STEPS" not in os.environ:
        os.environ["WAAM_BEAD_STEPS"] = "4000"
        args.skip_trends = True
        print("NOTE: --quick skips trend gates; use full n_steps for credibility.")

    baseline_job = load_job_config(CALIBRATE_JOB)
    cases = [
        ("calibrate (fitted)", CALIBRATE_JOB, None, "wall"),
        ("heldout_fast", HELDOUT_FAST_JOB, "smaller_pool", "distance"),
        ("heldout_hot", HELDOUT_HOT_JOB, "larger_pool", "wall"),
        ("heldout_macro2", HELDOUT_MACRO2_JOB, "larger_pool", "distance"),
    ]
    if args.with_bruno:
        cases.append(("heldout_bruno", HELDOUT_BRUNO_JOB, None, "distance"))

    results: dict[str, dict] = {}
    t0 = time.perf_counter()
    base_m = None
    for name, path, trend, mode in cases:
        job = load_job_config(path)
        if path != CALIBRATE_JOB:
            assert_physics_lock(baseline_job, job, label=name)
        print(f"\n=== {name} ===\n  job={path}")
        if path == CALIBRATE_JOB:
            m = run_job_metrics(path)
            base_m = m
        elif mode == "distance" and base_m is not None:
            m = run_job_metrics(path, equal_distance_m=base_m["travel_distance_mm"] / 1000.0)
        else:
            m = run_job_metrics(path, n_steps=base_m["n_steps"] if base_m else None)
        results[name] = m
        print(_fmt(m))
        print(
            f"  material={m['material_name']} status={m['material_status']}  "
            f"steps={m['n_steps']}  dist={m.get('travel_distance_mm', 0):.2f}mm"
        )
        if path in (HELDOUT_MACRO2_JOB, HELDOUT_BRUNO_JOB):
            assert_macrograph_prediction(m, job, label=name)

    base = results["calibrate (fitted)"]
    failed: list[str] = []
    if not args.skip_trends:
        for name, _path, trend, _mode in cases:
            if trend is None:
                continue
            try:
                assert_trend(base, results[name], trend)
                print(f"  trend OK ({trend})")
            except AssertionError as exc:
                print(f"  TREND FAIL: {exc}")
                failed.append(name)

    elapsed = time.perf_counter() - t0
    print(f"\n--- summary ({elapsed:.0f}s) ---")
    print(
        f"Fitted lock: η={base['eta']:.2f}  C_acc={base['recoil_accommodation']:.2f}  "
        f"calibrate W/D={base['pool_width_mm']:.2f}×{base['pool_depth_mm']:.2f} mm"
    )
    if base.get("macro_err_pct") is not None:
        print(f"Macrograph error (fitted case only): {base['macro_err_pct']:.1f}%")
    m2 = results.get("heldout_macro2")
    if m2:
        print(
            f"Macro2 slot predicted W/D={m2['pool_width_mm']:.2f}×{m2['pool_depth_mm']:.2f} mm "
            f"— paste measured values into {HELDOUT_MACRO2_JOB} reference when ready."
        )
    br = results.get("heldout_bruno")
    if br:
        print(
            f"Bruno surface-bead: pred W/h={br['pool_width_mm']:.2f}/{br['bead_height_mm']:.2f} mm "
            f"err={br.get('macro_err_pct', float('nan'))} — see docs/validation/PIONEER_BRUNO_DATASET.md"
        )
    print("Held-outs are predictions — do not retune Goldak/η/recoil to match them.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json}")

    if failed:
        print(f"Failed trends: {failed}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
