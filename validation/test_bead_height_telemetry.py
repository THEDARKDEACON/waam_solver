"""
test_bead_height_telemetry.py — bead_height_mm grows when metal is deposited.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import run_bead_travel


_JOB = "jobs/examples/bead_calibrate.yaml"


def run() -> None:
    init_taichi(backend="cpu")
    job = load_job_config(_JOB)
    twin = WAAMTwin(
        material=job["material"],
        nx=48, ny=28, nz=32, dx=4e-4,
        enable_vof=True,
        enable_bead_freeze=True,
        enable_wetting=True,
        max_tracers=20,
    )
    apply_job_to_twin(twin, job)
    twin.apply_plate_geometry(plate_thickness_mm=5.0, plate_size_mm=None)
    twin.reset()
    h0 = twin.get_telemetry()["bead_height_mm"]
    run_bead_travel(twin, 3500)
    telem = twin.get_telemetry()
    h1 = telem["bead_height_mm"]
    dep = telem["deposited_mass_g"]
    print(
        f"[bead_height_telemetry] h0={h0:.3f} h1={h1:.3f} mm  deposited={dep:.4f}g  "
        f"n_drops={telem['n_droplets_fired']}  ratio={telem['mass_balance_ratio']:.3f}"
    )
    if dep <= 0:
        raise AssertionError("No deposited mass recorded")
    if h1 <= h0 + 0.05:
        raise AssertionError("bead_height_mm should increase during welding")


if __name__ == "__main__":
    run()
    print("PASS")
