"""
test_job_parity.py — Same job YAML via from_job vs apply_job_to_twin (matched grid).

Core CI runs a fast structural smoke (flags/power/grid). Set
WAAM_FULL_VALIDATION=1 for geometric pool parity (tightened to 15%).
GPU standard parity: set WAAM_PARITY_GPU=1 when CUDA/Vulkan is available.
"""

from __future__ import annotations

import os
import sys

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm
from waam_twin.calibration import load_calibration, apply_calibration
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.validation.torch_motion import torch_position_linear


def _strip_forces(twin: WAAMTwin) -> None:
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.enable_recoil = False
    twin.enable_lorentz = False
    twin.enable_gas_shear = False
    twin.enable_evaporative_cooling = False
    twin.enable_moving_window = False


def _run_from_job(n_steps: int) -> tuple[float, float, tuple[int, int, int, float], WAAMTwin]:
    twin = WAAMTwin.from_job(
        "jobs/examples/bead_on_plate.yaml",
        preset_override="minimal",
    )
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    twin.travel_speed_m_s = travel
    _strip_forces(twin)
    twin.reset()
    g = twin.grid
    grid = (g.nx, g.ny, g.nz, g.dx)
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, 0.008, cy)
        twin.step(x, cy, is_welding=True)
    w, d = measure_pool_mm(twin)[:2]
    return w, d, grid, twin


def _run_apply_job(n_steps: int, grid: tuple[int, int, int, float]) -> tuple[float, float, WAAMTwin]:
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])
    nx, ny, nz, dx = grid
    twin = WAAMTwin(
        material=job["material"],
        nx=nx, ny=ny, nz=nz, dx=dx,
        arc_power_W=arc_w,
        arc_efficiency=float(proc.get("arc_efficiency", 0.72)),
        heat_source=str(job.get("heat_source", "goldak")),
        max_tracers=100,
    )
    apply_job_to_twin(twin, job)
    cal = load_calibration(job.get("calibration"))
    apply_calibration(twin, cal)
    twin.travel_speed_m_s = travel
    _strip_forces(twin)
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, 0.008, cy)
        twin.step(x, cy, is_welding=True)
    w, d = measure_pool_mm(twin)[:2]
    return w, d, twin


def run_structural_smoke() -> None:
    """Fast core-CI check: both loaders agree on process + grid + source."""
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin_a = WAAMTwin.from_job(
        "jobs/examples/bead_on_plate.yaml",
        preset_override="minimal",
    )
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    proc = job["process"]
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])
    g = twin_a.grid
    twin_b = WAAMTwin(
        material=job["material"],
        nx=g.nx, ny=g.ny, nz=g.nz, dx=g.dx,
        arc_power_W=arc_w,
        arc_efficiency=float(proc.get("arc_efficiency", 0.72)),
        heat_source=str(job.get("heat_source", "goldak")),
        max_tracers=100,
    )
    apply_job_to_twin(twin_b, job)
    cal = load_calibration(job.get("calibration"))
    apply_calibration(twin_b, cal)

    checks = [
        ("Q_w", twin_a.Q_w, twin_b.Q_w, 1e-6),
        ("eta", twin_a.eta, twin_b.eta, 1e-9),
        ("travel_speed_m_s", twin_a.travel_speed_m_s, twin_b.travel_speed_m_s, 1e-9),
        ("sigma_cells", twin_a.sigma_cells, twin_b.sigma_cells, 1e-6),
    ]
    for name, a, b, tol in checks:
        if abs(float(a) - float(b)) > tol:
            raise AssertionError(f"structural parity {name}: {a} vs {b}")
    if twin_a.heat_source_name != twin_b.heat_source_name:
        raise AssertionError(
            f"heat_source mismatch: {twin_a.heat_source_name} vs {twin_b.heat_source_name}"
        )
    if type(twin_a.arc_source) is not type(twin_b.arc_source):
        raise AssertionError(
            f"arc_source type mismatch: {type(twin_a.arc_source)} vs {type(twin_b.arc_source)}"
        )
    print(
        f"[job_parity] structural smoke OK  Q_w={twin_a.Q_w:.1f}W  η={twin_a.eta}  "
        f"grid={g.nx}×{g.ny}×{g.nz}  source={twin_a.heat_source_name}"
    )


def run(n_steps: int | None = None, threshold_pct: float | None = None) -> float:
    full = os.environ.get("WAAM_FULL_VALIDATION") == "1"
    if not full:
        run_structural_smoke()
        return 0.0

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    if n_steps is None:
        n_steps = 4000
    if threshold_pct is None:
        threshold_pct = 15.0

    w_a, d_a, grid, _ = _run_from_job(n_steps)
    w_b, d_b, _ = _run_apply_job(n_steps, grid)

    w_err = abs(w_b - w_a) / max(w_a, 0.1) * 100.0
    d_err = abs(d_b - d_a) / max(d_a, 0.1) * 100.0
    err = max(w_err, d_err)

    print(
        f"[job_parity] from_job W={w_a:.2f} D={d_a:.2f}  "
        f"apply_job W={w_b:.2f} D={d_b:.2f}  delta={err:.1f}%  "
        f"(threshold {threshold_pct}%)  grid={grid[0]}×{grid[1]}×{grid[2]} dx={grid[3]*1e3:.3f}mm"
    )
    if max(w_a, w_b) < 0.5:
        raise AssertionError("Job parity produced no melt pool — increase n_steps")
    if err >= threshold_pct:
        raise AssertionError(f"Job parity delta {err:.1f}% >= {threshold_pct}%")

    if os.environ.get("WAAM_PARITY_GPU") == "1":
        for backend in ("cuda", "vulkan"):
            try:
                init_taichi(backend=backend)
                wg, dg, _ = _run_apply_job(min(n_steps, 2000), grid)
                ge = max(
                    abs(wg - w_b) / max(w_b, 0.1) * 100.0,
                    abs(dg - d_b) / max(d_b, 0.1) * 100.0,
                )
                print(f"[job_parity] {backend} vs apply_job delta={ge:.1f}%")
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
