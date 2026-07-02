"""Deterministic wire mass / droplet sizing from process parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def wire_mass_flux_kg_s(twin: "WAAMTwin") -> float:
    """ṁ = ρ · (π d²/4) · wire_feed [kg/s]."""
    d = twin.wire_diameter_m
    area = math.pi * (d * 0.5) ** 2
    return twin.mat.rho * area * twin.wire_feed_m_s


def infer_transfer_mode(twin: "WAAMTwin") -> str:
    mode = str(getattr(twin, "droplet_transfer_mode", "auto")).strip().lower()
    if mode in {"globular", "spray", "pulsed"}:
        return mode
    I = float(getattr(twin, "welding_current_A", 0.0))
    if getattr(twin, "pulse_frequency_hz", 0.0) > 0.0:
        return "pulsed"
    if I < 180.0:
        return "globular"
    return "spray"


def base_droplet_period_s(twin: "WAAMTwin") -> float:
    freq = float(getattr(twin, "droplet_freq", 0.0))
    return 1.0 / max(freq, 1e-9) if freq > 0.0 else 1.0 / 50.0


def droplet_period_s(twin: "WAAMTwin") -> float:
    """Effective inter-drop period with deterministic transfer-mode modulation."""
    base = base_droplet_period_s(twin)
    mode = infer_transfer_mode(twin)
    if mode == "globular":
        mode_scale = 1.55
    elif mode == "spray":
        mode_scale = 0.72
    elif mode == "pulsed":
        pulse_hz = max(float(getattr(twin, "pulse_frequency_hz", 0.0)), 1e-9)
        mode_scale = max(0.25, min(4.0, (1.0 / pulse_hz) / base))
    else:
        mode_scale = 1.0
    idx = float(getattr(twin, "_n_droplets_fired", 0) + 1)
    jitter_amp = max(0.0, min(float(getattr(twin, "droplet_size_jitter", 0.0)), 0.35))
    jitter = 1.0 + jitter_amp * math.sin(0.73 * idx + 0.31)
    return max(base * 0.20, base * mode_scale * jitter)


def droplet_mass_for_interval_kg(twin: "WAAMTwin", dt_drop_s: float) -> float:
    return max(0.0, wire_mass_flux_kg_s(twin) * max(dt_drop_s, 0.0))


def droplet_mass_kg(twin: "WAAMTwin") -> float:
    if twin.droplet_freq <= 0:
        return 0.0
    return droplet_mass_for_interval_kg(twin, droplet_period_s(twin))


def droplet_radius_cells_from_mass_kg(twin: "WAAMTwin", m_drop: float) -> float:
    """
    Equivalent-sphere radius in grid cells from wire mass balance.

    Volume per drop: V = ṁ / (f_drop · ρ); r_cells from V / dx³.
    """
    g = twin.grid
    if m_drop <= 0:
        return 3.0
    vol = m_drop / twin.mat.rho
    n_cells = vol / (g.dx ** 3)
    r = (3.0 * n_cells / (4.0 * math.pi)) ** (1.0 / 3.0)
    return max(1.0, min(r, 10.0))


def droplet_radius_cells(twin: "WAAMTwin") -> float:
    return droplet_radius_cells_from_mass_kg(twin, droplet_mass_kg(twin))


def expected_deposited_mass_kg(twin: "WAAMTwin") -> float:
    """Integrated wire mass for elapsed sim time."""
    return wire_mass_flux_kg_s(twin) * twin._step_n * twin.grid.dt
