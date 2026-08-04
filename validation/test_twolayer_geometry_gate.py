"""
test_twolayer_geometry_gate.py — Remelt/HAZ extents on calibrate twolayer job.

Always checks physics lock vs bead_calibrate and that fusion/HAZ envelopes form.
Absolute remelt/HAZ vs experiment activates when reference.awaiting_measurement
is false and remelt_depth_mm is filled.
"""

from __future__ import annotations

import os
import sys

from waam_twin.job import load_job_config
from waam_twin.tools.multipass_report import TWOLAYER_JOB, run_twolayer_metrics
from waam_twin.validation.prediction import CALIBRATE_JOB, assert_physics_lock
from waam_twin.benchmark import pool_error_pct


def run(n_steps: int | None = None) -> dict:
    base = load_job_config(CALIBRATE_JOB)
    job = load_job_config(TWOLAYER_JOB)
    assert_physics_lock(base, job, label="twolayer_geometry")

    # Short path for core? No — this is FULL-only. Default uses model_reference.
    if n_steps is None:
        model = job.get("model_reference") or {}
        n_steps = int(os.environ.get("WAAM_TWOLAYER_STEPS", model.get("n_steps", 12000)))

    m = run_twolayer_metrics(n_steps=n_steps)
    ref = job.get("reference") or {}
    print(
        f"[twolayer_geom] fusion={m['fusion_width_mm']:.2f}×{m['fusion_depth_mm']:.2f} mm  "
        f"HAZ={m['haz_width_mm']:.2f}×{m['haz_depth_mm']:.2f} mm  "
        f"T_peak={m['haz_T_peak_K']:.0f}K  liq={m['n_liquid']}"
    )
    if m["haz_width_mm"] < 1.0 or m.get("n_haz_cells", 0) < 20:
        raise AssertionError(f"HAZ envelope too small: W={m['haz_width_mm']:.2f} mm")
    t_min = float(ref.get("haz_T_peak_min_K", 800))
    t_max = float(ref.get("haz_T_peak_max_K", 3200))
    if not (t_min <= m["haz_T_peak_K"] <= t_max):
        raise AssertionError(
            f"HAZ peak {m['haz_T_peak_K']:.0f}K outside [{t_min:.0f},{t_max:.0f}]"
        )
    # Full-budget runs should show a measurable fusion zone; short CI budgets may not.
    if n_steps >= 10000 and m["fusion_depth_mm"] < 0.4 and m["fusion_width_mm"] < 1.0:
        raise AssertionError(
            f"fusion zone undeveloped at {n_steps} steps: "
            f"{m['fusion_width_mm']:.2f}×{m['fusion_depth_mm']:.2f} mm"
        )
    if n_steps < 10000 and m["fusion_depth_mm"] < 0.4:
        print(
            f"[twolayer_geom] note: fusion depth {m['fusion_depth_mm']:.2f} mm "
            f"at short budget {n_steps} (HAZ OK)"
        )

    if not ref.get("awaiting_measurement", True) and ref.get("remelt_depth_mm") is not None:
        D_ref = float(ref["remelt_depth_mm"])
        W_ref = float(ref.get("haz_width_mm", m["haz_width_mm"]))
        err = pool_error_pct(m["haz_width_mm"], m["fusion_depth_mm"], W_ref, D_ref)
        thr = float(ref.get("tolerance_pct", 40.0))
        print(f"[twolayer_geom] vs experiment err={err:.1f}% (tol={thr}%)")
        if err >= thr:
            raise AssertionError(f"twolayer remelt/HAZ error {err:.1f}% >= {thr}%")
    else:
        print("[twolayer_geom] experiment PENDING — envelope gates only")
    return m


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
