"""
test_transfer_mode_detachment.py — Transfer mode changes detachment period and impact speed.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import deposition, weld_forces


def run() -> None:
    init_taichi(backend="cpu")
    base = dict(
        nx=20, ny=12, nz=12, dx=3e-4,
        wire_diameter_mm=1.2,
        droplet_freq_hz=44.0,
        travel_speed_m_s=0.008,
        welding_current_A=220.0,
    )
    glob = WAAMTwin(**base)
    glob.wire_feed_m_s = 8.0 / 60.0
    glob.droplet_transfer_mode = "globular"

    spray = WAAMTwin(**base)
    spray.wire_feed_m_s = 8.0 / 60.0
    spray.droplet_transfer_mode = "spray"

    pulsed = WAAMTwin(**base)
    pulsed.wire_feed_m_s = 8.0 / 60.0
    pulsed.droplet_transfer_mode = "pulsed"
    pulsed.pulse_frequency_hz = 90.0

    p_glob = deposition.droplet_period_s(glob)
    p_spray = deposition.droplet_period_s(spray)
    p_pulsed = deposition.droplet_period_s(pulsed)
    assert p_glob > p_spray, f"globular should detach slower than spray: {p_glob} vs {p_spray}"
    assert p_pulsed < p_glob, f"pulsed should not be slower than globular: {p_pulsed} vs {p_glob}"

    m_glob = deposition.droplet_mass_kg(glob)
    m_spray = deposition.droplet_mass_kg(spray)
    assert m_glob > m_spray, f"globular drops should be heavier: {m_glob} vs {m_spray}"

    v_glob = weld_forces.droplet_impact_velocity_m_s(glob, m_glob)
    v_spray = weld_forces.droplet_impact_velocity_m_s(spray, m_spray)
    assert v_spray >= v_glob, f"spray impact should be at least as fast: {v_spray} vs {v_glob}"
    print(
        f"[transfer_mode] periods g/s/p={p_glob:.4f}/{p_spray:.4f}/{p_pulsed:.4f} s  "
        f"masses g/s={m_glob*1e6:.2f}/{m_spray*1e6:.2f} mg  "
        f"v g/s={v_glob:.2f}/{v_spray:.2f} m/s"
    )


if __name__ == "__main__":
    run()
    print("PASS")
