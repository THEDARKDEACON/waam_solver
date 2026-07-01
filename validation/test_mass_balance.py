"""
test_mass_balance.py — Deposited volume tracks wire feed integral (deterministic).
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi


def run(tolerance: float = 0.35, n_steps: int = 8000) -> float:
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/bead_on_plate.yaml")
    twin = WAAMTwin(
        material=job["material"],
        nx=32, ny=20, nz=20, dx=3e-4,
        enable_vof=True,
        max_tracers=20,
    )
    apply_job_to_twin(twin, job)
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    travel = twin.travel_speed_m_s
    for step in range(n_steps):
        x = 0.004 + step * g.dt * travel
        twin.step(x, cy, is_welding=True)

    telem = twin.get_telemetry()
    ratio = telem["mass_balance_ratio"]
    print(
        f"[mass_balance] deposited={telem['deposited_mass_g']:.4f}g  "
        f"expected_drops={telem['expected_drop_mass_g']:.4f}g  "
        f"n_drops={telem['n_droplets_fired']}  ratio={ratio:.3f}"
    )
    if telem["deposited_mass_g"] <= 0:
        raise AssertionError("No metal deposited — droplet schedule may not have fired")
    if ratio < 1.0 - tolerance or ratio > 1.0 + tolerance:
        raise AssertionError(f"Mass balance ratio {ratio:.3f} outside ±{tolerance}")
    return ratio


if __name__ == "__main__":
    run()
    print("PASS")
