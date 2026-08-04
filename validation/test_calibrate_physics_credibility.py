"""
test_calibrate_physics_credibility.py — Full-tier calibrate: W/D + recoil + arc p + crown.

Runs the locked bead_calibrate job and asserts:
  - pool W/D within macrograph threshold
  - soft-onset CC recoil is active (f_recoil_max > 0)
  - Lin–Eagar arc pressure is in play
  - bead height grows; vapor cap not saturated
"""

from __future__ import annotations

import os
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.physics.weld_forces import arc_pressure_peak_pa, lin_eagar_peak_pa
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel


def run(n_steps: int | None = None, threshold_pct: float = 30.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job_path = "jobs/examples/bead_calibrate.yaml"
    job = load_job_config(job_path)
    ref = job.get("reference", {})
    model = job.get("model_reference") or {}
    W_ref = float(ref.get("pool_width_mm", 7.0))
    D_ref = float(ref.get("pool_depth_mm", 3.0))
    travel = float(job.get("process", {}).get("travel_speed_mm_s", 5.0)) / 1000.0
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000)))

    twin = WAAMTwin.from_job(job_path)
    if twin.physics_tier != "full":
        raise AssertionError(f"expected physics_tier=full, got {twin.physics_tier}")
    if twin.arc_pressure_model != "lin_eagar":
        raise AssertionError(f"expected lin_eagar, got {twin.arc_pressure_model}")
    p_arc = arc_pressure_peak_pa(twin)
    p_le = lin_eagar_peak_pa(twin.welding_current_A, twin.arc_sigma_m)
    if abs(p_arc - p_le) / max(p_le, 1.0) > 0.05:
        raise AssertionError(f"arc pressure {p_arc:.1f} Pa != lin_eagar {p_le:.1f} Pa")

    twin.travel_speed_m_s = travel
    twin.reset()
    h0 = twin.get_telemetry()["bead_height_mm"]
    n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    if n_liq < 1:
        raise AssertionError("No liquid cells")
    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    f_recoil = float(fd.get("f_recoil_max", 0.0))
    h1 = float(telem["bead_height_mm"])
    h_min = float(model.get("bead_height_mm_min", 1.5))

    print(
        f"[calibrate_physics] tier={twin.physics_tier}  p_arc={p_arc:.1f}Pa  "
        f"W={W_mm:.2f} D={D_mm:.2f} err={err:.1f}%  "
        f"T_peak={telem['peak_temp_C']:.0f}C  n_cap={telem.get('n_cells_at_vapor_cap', 0)}  "
        f"f_recoil={f_recoil:.3e}  h={h1:.2f}mm  "
        f"lorentz_unc={telem.get('lorentz_unconverged_streak', 0)}"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Pool geometry error {err:.1f}% >= {threshold_pct}%")
    if model.get("f_recoil_active", True) and f_recoil <= 0.0:
        raise AssertionError("soft-onset recoil expected f_recoil_max > 0")
    if telem.get("vapor_cap_saturated"):
        raise AssertionError("vapor cap saturated — raise evap_cooling_scale or lower power")
    if h1 < h_min:
        raise AssertionError(f"bead_height_mm {h1:.2f} < {h_min}")
    if h1 <= h0:
        raise AssertionError("bead height did not grow")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
