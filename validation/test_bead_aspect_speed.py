"""
test_bead_aspect_speed.py — Bead aspect h/w should decrease when travel speed increases.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import run_bead_travel


def _aspect_at_speed(travel_mm_s: float, n_steps: int = 2800) -> float:
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    twin = WAAMTwin(
        material=job["material"],
        nx=56, ny=28, nz=26, dx=3.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        enable_wetting=True,
        enable_hydrostatic_gravity=True,
        enable_bead_freeze=True,
        enable_deposition_momentum=True,
        heat_source="goldak",
        max_tracers=30,
    )
    apply_job_to_twin(twin, job)
    twin.travel_speed_m_s = travel_mm_s / 1000.0
    twin.reset()
    run_bead_travel(twin, n_steps)
    m = measure_bead_metrics(twin)
    return m["bead_aspect_hw"] if m["bead_height_mm"] > 0.05 else m["pool_aspect_dw"]


def run() -> None:
    init_taichi(backend="cpu")
    slow = _aspect_at_speed(8.0)
    fast = _aspect_at_speed(18.0)
    print(f"[bead_aspect_speed] aspect slow(8mm/s)={slow:.3f}  fast(18mm/s)={fast:.3f}")
    if fast >= slow * 0.98:
        raise AssertionError("Faster travel should not increase bead aspect ratio")


if __name__ == "__main__":
    run()
    print("PASS")
