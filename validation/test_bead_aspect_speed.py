"""
test_bead_aspect_speed.py — Faster travel should deposit a lower bead (less mass/length).

Compares equal travel distance at two speeds with deposition geometry isolated
from arc-force closures.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.platform import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import run_bead_travel, steps_for_travel


def _height_at_speed(travel_mm_s: float, distance_m: float = 0.012) -> tuple[float, float, float]:
    reset_taichi()
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/bead_calibrate.yaml")
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
    # Isolate deposition geometry from arc-force closures for this trend gate.
    twin.enable_lorentz = False
    twin.enable_gas_shear = False
    twin.enable_recoil = False
    twin.enable_evaporative_cooling = False
    twin.arc_pressure = 0.0
    twin.arc_pressure_model = "constant"
    twin.travel_speed_m_s = travel_mm_s / 1000.0
    twin.reset()
    n_steps = steps_for_travel(distance_m, twin.travel_speed_m_s, twin.grid.dt)
    run_bead_travel(twin, n_steps)
    m = measure_bead_metrics(twin)
    h = m["bead_height_mm"] if m["bead_height_mm"] > 0.05 else m["pool_depth_mm"]
    w = m["bead_width_mm"] if m["bead_width_mm"] > 0.05 else m["pool_width_mm"]
    aspect = h / max(w, 0.1)
    return h, w, aspect


def run() -> None:
    init_taichi(backend="cpu")
    h_slow, w_slow, a_slow = _height_at_speed(8.0)
    h_fast, w_fast, a_fast = _height_at_speed(18.0)
    print(
        f"[bead_aspect_speed] slow(8mm/s) h={h_slow:.2f} w={w_slow:.2f} a={a_slow:.3f}  "
        f"fast(18mm/s) h={h_fast:.2f} w={w_fast:.2f} a={a_fast:.3f}"
    )
    # Equal distance: faster travel ⇒ less mass/length ⇒ lower or equal height.
    if h_fast > h_slow + 0.15:
        raise AssertionError(
            f"Faster travel should not raise bead height ({h_fast:.2f} > {h_slow:.2f})"
        )
    if a_fast > a_slow + 0.05:
        raise AssertionError(
            f"Faster travel should not increase bead aspect ({a_fast:.3f} > {a_slow:.3f})"
        )


if __name__ == "__main__":
    run()
    print("PASS")
