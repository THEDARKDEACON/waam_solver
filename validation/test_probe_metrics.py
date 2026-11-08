"""
test_probe_metrics.py — probe sampling interval, thermocouple model, cycle
metrics and the job-block round trip.

  - thermal_cycle_metrics on an analytic cycle: peak, t8/5 (exact for a
    linear cool-down), time above solidus; a reheated second cycle (wall)
    gives the t8/5 of the last pass.
  - ProbeRecorder.sample_every_ms records on the interval, not every step.
  - time_constant_ms lags a temperature step by 1 − e^(−t/τ).
  - bead_diameter_mm averages over the bead (a uniform field is unchanged).
  - position_tolerance_mm brackets the point value.
  - to_job_block → from_job_list keeps names, positions and sensor keys.
"""

from __future__ import annotations

import math

import numpy as np

from waam_twin.export.probe_metrics import thermal_cycle_metrics

C0 = 273.15


def _metrics() -> None:
    # Heat 25 → 1500 °C in 100 ms, then cool linearly at 300 K/s.
    t = np.arange(0.0, 6000.0, 1.0)
    T = np.where(t < 100, 25 + 14.75 * t, 1500 - 0.3 * (t - 100)) + C0
    T = np.maximum(T, 25 + C0)
    m = thermal_cycle_metrics(t, T, 1450 + C0)
    assert abs(m["peak_C"] - 1500) < 0.5, m
    assert abs(m["t85_s"] - 1.0) < 2e-3, m  # 300 K over 300 K/s
    assert abs(m["cool_700_K_s"] - 300.0) < 1.0, m
    assert 160 < m["above_solidus_ms"] < 175, m  # ~3.4 ms up + 166.7 ms down
    # Second, cooler pass (next layer): reheat to 900 °C, cool at 150 K/s.
    t2 = np.arange(6000.0, 12000.0, 1.0)
    T2 = np.where(t2 < 6100, T[-1] + (900 + C0 - T[-1]) * (t2 - 6000) / 100, 900 + C0 - 0.15 * (t2 - 6100))
    m2 = thermal_cycle_metrics(np.r_[t, t2], np.r_[T, T2], 1450 + C0)
    assert abs(m2["t85_s"] - 2.0) < 4e-3, m2  # last pass sets it
    assert math.isnan(thermal_cycle_metrics(t[:150], T[:150], 1450 + C0)["t85_s"])
    print(f"[probe_metrics] t8/5 {m['t85_s']:.3f} s, wall reheat {m2['t85_s']:.3f} s")


def _recorder() -> None:
    from waam_twin import WAAMTwin
    from waam_twin.export.probes import ProbeRecorder
    from waam_twin.runtime import init_taichi

    init_taichi()
    twin = WAAMTwin(nx=16, ny=12, nz=12, dx=4e-4, max_tracers=4, enable_vof=True)
    twin.reset()
    g = twin.grid
    dt_ms = g.dt * 1000.0
    z_mm = 0.5 * int(twin.nz_solid) * 0.4  # mid-plate
    rec = ProbeRecorder(sample_every_ms=10 * dt_ms)
    rec.add_world_mm(3.2, 2.4, z_mm, twin, "pt")
    rec.add_world_mm(3.2, 2.4, z_mm, twin, "tc")
    tc = rec.probes[-1]
    tc.time_constant_s = 20 * g.dt
    tc.bead_diameter_m = 0.8e-3
    tc.position_tolerance_m = 0.4e-3
    g.T.fill(500.0)
    for step in range(0, 25):
        twin._step_n = step
        rec.record_step(twin)
    assert [r["step"] for r in rec._rows] == [0, 10, 20], [r["step"] for r in rec._rows]
    r = rec._rows[-1]
    assert abs(r["tc_T_tc_K"] - 500.0) < 1e-3, r  # uniform field: bead average unchanged
    assert r["tc_T_lo_K"] == r["tc_T_hi_K"] == 500.0
    g.T.fill(1500.0)  # step change
    for step in range(25, 45):
        twin._step_n = step
        rec.record_step(twin)
    # The step is first sampled at step 30; by step 40 the filter has seen
    # 20 steps (= τ) of it: 1 − e^(−1) of the way.
    want = 500.0 + 1000.0 * (1.0 - math.exp(-20 / 20))
    got = rec._rows[-1]["tc_T_tc_K"]
    assert abs(got - want) < 0.5, (got, want)
    assert rec._rows[-1]["pt_T_K"] == 1500.0
    s = {m["name"]: m for m in rec.summary(1700.0)}
    assert abs(s["pt"]["peak_C"] - (1500 - C0)) < 0.1 and s["tc"]["peak_C"] < s["pt"]["peak_C"]

    from waam_twin.export.probes import ProbeRecorder as PR
    block = rec.to_job_block()
    back = PR.from_job_list(block, twin)
    assert [p.name for p in back.probes] == ["pt", "tc"]
    assert abs(back.sample_every_ms - rec.sample_every_ms) < 1e-12
    b = back.probes[1]
    assert abs(b.z_m - tc.z_m) < 1e-9 and abs(b.bead_diameter_m - 0.8e-3) < 1e-12
    assert abs(b.time_constant_s - tc.time_constant_s) < 1e-9
    assert PR.from_job_list([{"name": "a", "x_mm": 1, "y_mm": 1, "z_mm": 1}], twin).sample_every_ms == 1.0
    print(f"[probe_metrics] interval + thermocouple lag ok (tc {got:.1f} K vs {want:.1f} K)")


def run() -> None:
    _metrics()
    _recorder()


if __name__ == "__main__":
    run()
    print("PASS")
