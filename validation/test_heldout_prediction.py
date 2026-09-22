"""
test_heldout_prediction.py — Physics lock + held-out trend / macrograph gates.

1. Always: held-out YAMLs must not retune Goldak/η/recoil vs bead_calibrate.
2. Under WAAM_HELDOUT_VALIDATION=1: run baseline + held-outs,
   assert directional trends, and absolute macrograph error when reference is filled.
"""

from __future__ import annotations

import os
import sys

from waam_twin.job import load_job_config
from waam_twin.validation.prediction import (
    CALIBRATE_JOB,
    HELDOUT_BRUNO_JOB,
    HELDOUT_FAST_JOB,
    HELDOUT_HOT_JOB,
    HELDOUT_MACRO2_JOB,
    PIONEER_WALL_JOB,
    assert_macrograph_prediction,
    assert_physics_lock,
    assert_trend,
    run_job_metrics,
)


def run(*, run_sims: bool | None = None) -> dict:
    if run_sims is None:
        run_sims = os.environ.get("WAAM_HELDOUT_VALIDATION") == "1"

    base_job = load_job_config(CALIBRATE_JOB)
    for path, label in (
        (HELDOUT_FAST_JOB, "heldout_fast"),
        (HELDOUT_HOT_JOB, "heldout_hot"),
        (HELDOUT_MACRO2_JOB, "heldout_macro2"),
        (HELDOUT_BRUNO_JOB, "heldout_bruno"),
        (PIONEER_WALL_JOB, "pioneer_m1_wall"),
    ):
        assert_physics_lock(base_job, load_job_config(path), label=label)

    mat = str(base_job.get("material", ""))
    if "validated" not in mat or "ER70S-6" not in mat:
        raise AssertionError(f"calibrate should use validated ER70S-6, got {mat!r}")
    if base_job.get("calibration") not in (None, "", "null"):
        raise AssertionError(
            "calibrate lock uses job-embedded η — calibration overlay should be null"
        )

    print(
        f"[heldout_lock] fast+hot+macro2+bruno+pioneer match calibrate Goldak/η/recoil lock  "
        f"material={mat}"
    )
    bruno_ref = load_job_config(HELDOUT_BRUNO_JOB).get("reference") or {}
    if not bruno_ref.get("bead_width_mm"):
        raise AssertionError("Bruno surface reference missing bead_width_mm")
    pioneer_ref = load_job_config(PIONEER_WALL_JOB).get("reference") or {}
    if pioneer_ref.get("awaiting_measurement") or pioneer_ref.get("remelt_depth_mm") is None:
        raise AssertionError("PIONEER wall remelt_depth_mm should be filled from macros")
    print(
        f"[heldout_lock] Bruno surface W/H={bruno_ref.get('bead_width_mm')}/"
        f"{bruno_ref.get('bead_height_mm')}  "
        f"PIONEER remelt={pioneer_ref.get('remelt_depth_mm')} mm "
        f"wall_w={pioneer_ref.get('wall_width_mm')} mm"
    )
    if not run_sims:
        print("[heldout_lock] sim trends skipped (set WAAM_HELDOUT_VALIDATION=1)")
        return {"lock_only": True}

    n_steps = int(os.environ.get(
        "WAAM_BEAD_STEPS",
        (base_job.get("model_reference") or {}).get("n_steps", 8000),
    ))
    base = run_job_metrics(CALIBRATE_JOB, n_steps=n_steps)
    dist_m = base["travel_distance_mm"] / 1000.0
    fast = run_job_metrics(HELDOUT_FAST_JOB, equal_distance_m=dist_m)
    hot = run_job_metrics(HELDOUT_HOT_JOB, n_steps=n_steps)
    # Macro2 is slower travel → equal distance (more steps) for fair HIL.
    macro2 = run_job_metrics(HELDOUT_MACRO2_JOB, equal_distance_m=dist_m)
    macro2_job = load_job_config(HELDOUT_MACRO2_JOB)
    bruno = run_job_metrics(HELDOUT_BRUNO_JOB, equal_distance_m=dist_m)
    bruno_job = load_job_config(HELDOUT_BRUNO_JOB)

    print(
        f"[heldout] calibrate W={base['pool_width_mm']:.2f} D={base['pool_depth_mm']:.2f} "
        f"err={base.get('macro_err_pct', float('nan')):.1f}%  dist={base['travel_distance_mm']:.2f}mm  "
        f"fast={fast['pool_width_mm']:.2f}/{fast['pool_depth_mm']:.2f}  "
        f"hot={hot['pool_width_mm']:.2f}/{hot['pool_depth_mm']:.2f}  "
        f"macro2={macro2['pool_width_mm']:.2f}/{macro2['pool_depth_mm']:.2f}  "
        f"bruno={bruno['pool_width_mm']:.2f}/{bruno['bead_height_mm']:.2f}"
    )
    if base.get("macro_err_pct", 100.0) >= 30.0:
        raise AssertionError(
            f"fitted baseline macrograph error {base['macro_err_pct']:.1f}% >= 30%"
        )
    assert_trend(base, fast, "smaller_pool")
    assert_trend(base, hot, "larger_pool")
    assert_trend(base, macro2, "larger_pool")
    assert_macrograph_prediction(macro2, macro2_job, label="heldout_macro2")
    assert_macrograph_prediction(bruno, bruno_job, label="heldout_bruno", threshold_pct=50.0)
    return {
        "calibrate": base,
        "fast": fast,
        "hot": hot,
        "macro2": macro2,
        "bruno": bruno,
    }


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
