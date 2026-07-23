"""
test_bead_macrograph_gate.py — Full physics vs experimental reference (standard preset).

Set WAAM_BEAD_VALIDATION=1 in run_all. Tolerance is intentionally strict (40%).
"""

from __future__ import annotations

import os

from waam_twin import WAAMTwin
from waam_twin.benchmark import bead_error_pct, measure_bead_metrics
from waam_twin.job import load_job_config
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(threshold_pct: float | None = None) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    if threshold_pct is None:
        threshold_pct = float(os.environ.get("WAAM_BEAD_TOLERANCE_PCT", "40"))

    job = load_job_config("jobs/examples/bead_calibrate.yaml")
    ref = job.get("reference", {})
    W_ref = float(ref.get("pool_width_mm", 7.0))
    D_ref = float(ref.get("pool_depth_mm", 3.0))

    twin = WAAMTwin.from_job("jobs/examples/bead_calibrate.yaml", preset_override="standard")
    twin.reset()
    n_steps_env = os.environ.get("WAAM_BEAD_STEPS")
    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(
        twin, job, n_steps=int(n_steps_env) if n_steps_env else None
    )
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    m = measure_bead_metrics(twin)
    err = bead_error_pct(m, W_ref, D_ref)
    print(
        f"[bead_macrograph] pool W={m['pool_width_mm']:.2f} D={m['pool_depth_mm']:.2f} mm  "
        f"bead h={m['bead_height_mm']:.2f} w={m['bead_width_mm']:.2f} mm  "
        f"dep={twin.get_telemetry()['deposited_mass_g']:.3f}g  err={err:.1f}%"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Macrograph bead error {err:.1f}% >= {threshold_pct}%")
    return err


if __name__ == "__main__":
    run()
    print("PASS")
