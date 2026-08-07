"""
test_intensive_calibrate_soak.py — Full-tier calibrate job stability soak.

Runs bead_calibrate for an extended budget and asserts:
  - no NaN/Inf in T
  - mass balance within tolerance
  - pool W/D still within macrograph envelope
  - vapor cap not saturated; soft-onset recoil remains active
"""

from __future__ import annotations

import os
import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.runtime import init_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(n_steps: int | None = None, threshold_pct: float = 35.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job_path = "jobs/examples/bead_calibrate.yaml"
    job = load_job_config(job_path)
    ref = job.get("reference", {})
    model = job.get("model_reference") or {}
    if n_steps is None:
        # Default: model reference + extra soak tail
        # Default soak = model reference length (extra tail via WAAM_SOAK_STEPS)
        base = int(model.get("n_steps", 8000))
        n_steps = int(os.environ.get("WAAM_SOAK_STEPS", str(base)))

    twin = WAAMTwin.from_job(job_path)
    twin.enable_force_diagnostics = True
    twin.travel_speed_m_s = float(job["process"]["travel_speed_mm_s"]) / 1000.0
    twin.reset()

    planned, x0, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    # Mid-run NaN probe: travel in chunks
    chunk = max(planned // 4, 1000)
    done = 0
    x = x0
    g = twin.grid
    while done < planned:
        n = min(chunk, planned - done)
        run_bead_travel(
            twin, n,
            x_start_m=x, y_m=y_m, direction_x=dir_x,
        )
        x = x + dir_x * twin.travel_speed_m_s * g.dt * n
        done += n
        if not np.isfinite(g.T.to_numpy()).all():
            raise AssertionError(f"NaN/Inf in T after {done} soak steps")

    W, D, n_liq = measure_pool_mm(twin)
    err = pool_error_pct(
        W, D,
        float(ref.get("pool_width_mm", 7.0)),
        float(ref.get("pool_depth_mm", 3.0)),
    )
    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    bal = float(telem.get("mass_balance_ratio", 0.0))

    print(
        f"[intensive_calibrate_soak] steps={planned}  W={W:.2f} D={D:.2f} "
        f"err={err:.1f}%  bal={bal:.2f}  T={telem['peak_temp_C']:.0f}C  "
        f"n_cap={telem.get('n_cells_at_vapor_cap', 0)}  "
        f"f_recoil={float(fd.get('f_recoil_max', 0)):.2e}  "
        f"Lz_streak={telem.get('lorentz_unconverged_streak', 0)}"
    )
    if n_liq < 1:
        raise AssertionError("soak ended with no liquid")
    if err >= threshold_pct:
        raise AssertionError(f"soak pool error {err:.1f}% >= {threshold_pct}%")
    if not (0.7 <= bal <= 1.4):
        raise AssertionError(f"mass balance {bal:.2f} outside [0.7, 1.4]")
    if telem.get("vapor_cap_saturated"):
        raise AssertionError("vapor cap saturated during calibrate soak")
    if float(fd.get("f_recoil_max", 0.0)) <= 0.0:
        raise AssertionError("recoil inactive at end of soak")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
