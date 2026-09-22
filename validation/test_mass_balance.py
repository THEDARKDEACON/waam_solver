"""
test_mass_balance.py — Deposited volume tracks wire feed integral (deterministic).
Also: expected wire mass freezes when the arc is off (cooling dwell).
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config, apply_job_to_twin
from waam_twin.runtime import init_taichi
from waam_twin.physics.deposition_balance import wire_mass_flux_kg_s


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
    twin.strict_mode = False  # this test uses a reduced grid + ±35% wire-mass band
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
        f"expected_wire={telem['expected_wire_mass_g']:.4f}g  "
        f"expected_drops={telem['expected_drop_mass_g']:.4f}g  "
        f"n_drops={telem['n_droplets_fired']}  overflow={telem.get('deposition_overflow_count', 0)}  "
        f"ratio={ratio:.3f}  nz_solid={twin.nz_solid}/{g.nz}"
    )
    if telem["n_droplets_fired"] < 1:
        raise AssertionError("Droplet schedule never fired — check wire_feed / droplet_freq")
    if telem["deposited_mass_g"] <= 0:
        raise AssertionError("No metal deposited — droplet schedule may not have fired")
    if ratio < 1.0 - tolerance or ratio > 1.0 + tolerance:
        raise AssertionError(
            f"Mass balance ratio {ratio:.3f} (deposited / ṁ·t_weld) outside ±{tolerance}"
        )
    drop_g = float(telem["expected_drop_mass_g"])
    if drop_g > 1e-9:
        drop_ratio = float(telem["deposited_mass_g"]) / drop_g
        if drop_ratio < 0.4 or drop_ratio > 2.5:
            raise AssertionError(
                f"Drop-count diagnostic ratio {drop_ratio:.3f} wildly off "
                f"(deposited vs n×instantaneous drop mass)"
            )

    # Arc-off dwell must not inflate the expected-wire ledger (HUD "wire" mass).
    wire_on = float(telem["expected_wire_mass_g"])
    dep_on = float(telem["deposited_mass_g"])
    t_weld = float(telem["welding_time_s"])
    expected_from_weld_t = wire_mass_flux_kg_s(twin) * t_weld * 1000.0
    if abs(wire_on - expected_from_weld_t) > 1e-3:
        raise AssertionError(
            f"expected_wire_mass_g={wire_on:.4f} != ṁ·t_weld={expected_from_weld_t:.4f}"
        )
    for _ in range(2000):
        twin.step(x, cy, is_welding=False)
    telem2 = twin.get_telemetry()
    if abs(float(telem2["expected_wire_mass_g"]) - wire_on) > 1e-6:
        raise AssertionError(
            f"wire mass grew during arc-off: {wire_on:.4f} → {telem2['expected_wire_mass_g']:.4f} g"
        )
    if abs(float(telem2["deposited_mass_g"]) - dep_on) > 1e-6:
        raise AssertionError(
            f"deposited mass changed during arc-off: {dep_on:.4f} → {telem2['deposited_mass_g']:.4f} g"
        )
    if abs(float(telem2["welding_time_s"]) - t_weld) > 1e-9:
        raise AssertionError("welding_time_s advanced while is_welding=False")
    print(
        f"[mass_balance] arc-off freeze OK  wire={wire_on:.4f}g  "
        f"t_weld={t_weld:.4f}s  (sim continued {2000 * g.dt * 1000:.0f} ms)"
    )
    return ratio


if __name__ == "__main__":
    run()
    print("PASS")
