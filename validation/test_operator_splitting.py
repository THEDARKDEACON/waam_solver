"""Quantify first-order Lie splitting: peak-T change under dt/2.

Thermal advection uses previous-step velocity; error is O(Δt). A short weld
at dt and dt/2 should keep relative peak-temperature change within a modest band.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi


def _peak_T(twin: WAAMTwin, n_steps: int) -> float:
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x = 0.004 + step * g.dt * 0.01
        twin.step(x, cy, is_welding=True)
    return float(twin.get_telemetry()["peak_temp_K"])


def run(max_rel: float = 0.08) -> None:
    init_taichi(backend="cpu")
    kw = dict(
        nx=20, ny=16, nz=16, dx=4e-4,
        enable_vof=False,
        enable_enthalpy_cap=True,
        heat_source="goldak",
        max_tracers=8,
        C_darcy=0.0,
        use_srt=True,
        use_variable_tau=False,
        dt_scale=1.0,
        arc_power_W=2000.0,
        arc_efficiency=0.8,
    )
    twin_a = WAAMTwin(**kw)
    twin_a.wire_feed_m_s = 0.05
    T_a = _peak_T(twin_a, n_steps=120)

    twin_b = WAAMTwin(**{**kw, "dt_scale": 0.5})
    twin_b.wire_feed_m_s = 0.05
    # Twice the steps so physical time matches.
    T_b = _peak_T(twin_b, n_steps=240)

    denom = max(abs(T_a), abs(T_b), 1.0)
    rel = abs(T_a - T_b) / denom
    print(
        f"[operator_splitting] T(dt)={T_a:.1f}K  T(dt/2)={T_b:.1f}K  "
        f"rel={rel:.4f}  gate={max_rel}"
    )
    if T_a < 400.0 or T_b < 400.0:
        raise AssertionError("splitting probe under-heated — raise steps/power")
    if rel > max_rel:
        raise AssertionError(
            f"splitting sensitivity too large: rel={rel:.4f} > {max_rel}"
        )


if __name__ == "__main__":
    run()
    print("PASS")
