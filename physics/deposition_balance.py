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


def droplet_mass_kg(twin: "WAAMTwin") -> float:
    if twin.droplet_freq <= 0:
        return 0.0
    return wire_mass_flux_kg_s(twin) / twin.droplet_freq


def droplet_radius_cells(twin: "WAAMTwin") -> float:
    """
    Equivalent-sphere radius in grid cells from wire mass balance.

    Volume per drop: V = ṁ / (f_drop · ρ); r_cells from V / dx³.
    """
    g = twin.grid
    m_drop = droplet_mass_kg(twin)
    if m_drop <= 0:
        return 3.0
    vol = m_drop / twin.mat.rho
    n_cells = vol / (g.dx ** 3)
    r = (3.0 * n_cells / (4.0 * math.pi)) ** (1.0 / 3.0)
    return max(1.0, min(r, 10.0))


def expected_deposited_mass_kg(twin: "WAAMTwin") -> float:
    """Integrated wire mass for elapsed sim time."""
    return wire_mass_flux_kg_s(twin) * twin._step_n * twin.grid.dt
