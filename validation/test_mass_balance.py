"""
test_mass_balance.py — Deposited volume tracks wire feed integral (deterministic).
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi


_JOB = "jobs/examples/bead_calibrate.yaml"


def run(tolerance: float = 0.35, n_steps: int = 8000) -> float:
    init_taichi(backend="cpu")
    job = load_job_config(_JOB)
    # Small grid with explicit thin plate so air remains for feed_wire_surface.
    twin = WAAMTwin(
        material=job["material"],
        nx=40, ny=24, nz=28, dx=3.5e-4,
        enable_vof=True,
        max_tracers=20,
    )
    apply_job_to_twin(twin, job)
    twin.apply_plate_geometry(plate_thickness_mm=4.0, plate_size_mm=None)
    twin.reset()
    g = twin.grid
    assert twin.nz_solid <= g.nz - 6, (
        f"need air headroom for deposition, got nz_solid={twin.nz_solid} nz={g.nz}"
    )
    cy = (g.ny // 2) * g.dx
    travel = twin.travel_speed_m_s
    for step in range(n_steps):
        x = 0.005 + step * g.dt * travel
        twin.step(x, cy, is_welding=True)

    telem = twin.get_telemetry()
    ratio = telem["mass_balance_ratio"]
    print(
        f"[mass_balance] deposited={telem['deposited_mass_g']:.4f}g  "
        f"expected_drops={telem['expected_drop_mass_g']:.4f}g  "
        f"n_drops={telem['n_droplets_fired']}  overflow={telem.get('deposition_overflow_count', 0)}  "
        f"ratio={ratio:.3f}  nz_solid={twin.nz_solid}/{g.nz}"
    )
    if telem["n_droplets_fired"] < 1:
        raise AssertionError("Droplet schedule never fired — check wire_feed / droplet_freq")
    if telem["deposited_mass_g"] <= 0:
        raise AssertionError("No metal deposited — droplet schedule may not have fired")
    if ratio < 1.0 - tolerance or ratio > 1.0 + tolerance:
        raise AssertionError(f"Mass balance ratio {ratio:.3f} outside ±{tolerance}")
    return ratio


if __name__ == "__main__":
    run()
    print("PASS")
