"""
test_stickout_preheat.py — Stick-out resistance raises droplet entry temperature.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics.electrical_stickout import droplet_entry_temperature_K


def run() -> None:
    init_taichi(backend="cpu")
    base = WAAMTwin(nx=16, ny=10, nz=12, dx=3e-4, enable_ctwd=False)
    hot = WAAMTwin(
        nx=16, ny=10, nz=12, dx=3e-4,
        enable_ctwd=True, stickout_mm=20.0, welding_current_A=180.0,
    )
    hot.wire_feed_m_s = 8.0 / 60.0
    T_base = droplet_entry_temperature_K(base)
    T_hot = droplet_entry_temperature_K(hot)
    assert T_hot > T_base + 0.5, f"expected preheat lift: {T_base} -> {T_hot}"
    print(f"[stickout_preheat] T_base={T_base:.0f}K T_hot={T_hot:.0f}K")


if __name__ == "__main__":
    run()
    print("PASS")
