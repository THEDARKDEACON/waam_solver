"""
test_calibrated_pool.py — Reproduce fitted calibration W/D within ±30% on benchmark grid.
"""

from __future__ import annotations

import pathlib
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.calibration import load_calibration, apply_calibration
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.validation.torch_motion import torch_position_linear


def _fit_target(job: dict) -> tuple[float, float, int]:
    """Read W/D/n_steps from calibration fit_metrics, else model_reference."""
    cal_path = job.get("calibration")
    if cal_path:
        try:
            import yaml
            with open(pathlib.Path(cal_path)) as f:
                data = yaml.safe_load(f) or {}
            metrics = data.get("fit_metrics", {})
            if metrics.get("pool_width_mm") is not None:
                return (
                    float(metrics["pool_width_mm"]),
                    float(metrics.get("pool_depth_mm", 2.8)),
                    int(metrics.get("n_steps", 2000)),
                )
        except Exception:
            pass
    model = job.get("model_reference", {})
    return (
        float(model.get("pool_width_mm", 4.2)),
        float(model.get("pool_depth_mm", 5.4)),
        int(model.get("n_steps", 2000)),
    )


def run(n_steps: int | None = None, threshold_pct: float = 30.0) -> float:
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    W_ref, D_ref, n_fit = _fit_target(job)
    n_steps = n_steps or n_fit
    macro = job.get("reference", {})
    W_macro = float(macro.get("pool_width_mm", 7.0))
    D_macro = float(macro.get("pool_depth_mm", 2.8))
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])

    cal = load_calibration(job.get("calibration"))
    twin = WAAMTwin(
        material=job["material"],
        nx=88, ny=44, nz=44, dx=3e-4,
        arc_power_W=arc_w,
        arc_efficiency=proc.get("arc_efficiency", 0.72),
        max_tracers=100,
    )
    apply_job_to_twin(twin, job)
    apply_calibration(twin, cal)
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.reset()

    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, 0.015, cy)
        twin.step(x, cy, is_welding=True)

    W_mm, D_mm, _ = measure_pool_mm(twin)
    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    macro_err = pool_error_pct(W_mm, D_mm, W_macro, D_macro)
    print(
        f"[calibrated_pool] W={W_mm:.2f} D={D_mm:.2f}  "
        f"fit_err={err:.1f}%  macro_err={macro_err:.1f}%  (threshold {threshold_pct}%)"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Calibrated pool error {err:.1f}% >= {threshold_pct}%")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
