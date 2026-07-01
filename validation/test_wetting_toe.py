"""
test_wetting_toe.py — Toe angle estimate tracks contact_angle_deg order-of-magnitude.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.physics.bead_geometry import estimate_toe_angle_deg
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import run_bead_travel


def run() -> None:
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    twin = WAAMTwin(
        material=job["material"],
        nx=40, ny=22, nz=22, dx=3.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        enable_wetting=True,
        max_tracers=10,
    )
    apply_job_to_twin(twin, job)
    twin.contact_angle_deg = 80.0
    twin.theta_rad = 80.0 * 3.14159 / 180.0
    twin.reset()
    run_bead_travel(twin, 1800)
    toe = estimate_toe_angle_deg(twin, twin.grid)
    target = twin.contact_angle_deg
    print(f"[wetting_toe] toe={toe:.1f}°  target={target:.1f}°")
    if abs(toe - target) > 35.0:
        raise AssertionError(f"Toe angle {toe:.1f}° too far from contact angle {target:.1f}°")


if __name__ == "__main__":
    run()
    print("PASS")
