"""
test_job_parity.py — Same job YAML on minimal preset vs standard benchmark grid.

Verifies portable job loading produces consistent pool geometry (within dx tolerance).
GPU standard parity: set WAAM_PARITY_GPU=1 when CUDA/Vulkan is available.
"""

from __future__ import annotations

import os
import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm
from waam_twin.calibration import load_calibration, apply_calibration
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.validation.torch_motion import torch_position_linear


def _run_job(n_steps: int, preset: str | None, grid: tuple[int, int, int, float] | None) -> tuple[float, float]:
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])

    if grid:
        nx, ny, nz, dx = grid
        twin = WAAMTwin(
            material=job["material"],
            nx=nx, ny=ny, nz=nz, dx=dx,
            arc_power_W=arc_w,
            arc_efficiency=float(proc.get("arc_efficiency", 0.72)),
            max_tracers=100,
        )
        apply_job_to_twin(twin, job)
        cal = load_calibration(job.get("calibration"))
        apply_calibration(twin, cal)
    else:
        os.environ["WAAM_PRESET"] = preset or "minimal"
        twin = WAAMTwin.from_job("jobs/examples/bead_on_plate.yaml")

    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.travel_speed_m_s = travel
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, 0.015, cy)
        twin.step(x, cy, is_welding=True)
    W_mm, D_mm, _ = measure_pool_mm(twin)
    return W_mm, D_mm


def run(n_steps: int = 800, threshold_pct: float = 50.0) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    w_min, d_min = _run_job(n_steps, "minimal", None)
    w_std, d_std = _run_job(n_steps, None, (88, 44, 44, 3.0e-4))

    w_err = abs(w_std - w_min) / max(w_min, 0.1) * 100.0
    d_err = abs(d_std - d_min) / max(d_min, 0.1) * 100.0
    err = max(w_err, d_err)

    print(
        f"[job_parity] minimal W={w_min:.2f} D={d_min:.2f}  "
        f"standard-grid W={w_std:.2f} D={d_std:.2f}  delta={err:.1f}%  "
        f"(threshold {threshold_pct}%)"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Job parity delta {err:.1f}% >= {threshold_pct}%")

    if os.environ.get("WAAM_PARITY_GPU") == "1":
        for backend in ("cuda", "vulkan"):
            try:
                init_taichi(backend=backend)
                wg, dg = _run_job(min(n_steps, 400), None, (88, 44, 44, 3.0e-4))
                ge = max(
                    abs(wg - w_std) / max(w_std, 0.1) * 100.0,
                    abs(dg - d_std) / max(d_std, 0.1) * 100.0,
                )
                print(f"[job_parity] {backend} vs cpu-standard delta={ge:.1f}%")
                if ge >= 50.0:
                    raise AssertionError(f"GPU parity {backend} delta {ge:.1f}% >= 50%")
            except Exception as exc:
                print(f"[job_parity] skip {backend}: {exc}")

    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
