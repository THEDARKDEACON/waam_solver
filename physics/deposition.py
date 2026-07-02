"""Wire deposition and droplet momentum."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import kernels
from .deposition_balance import (
    droplet_mass_for_interval_kg,
    droplet_mass_kg,
    droplet_period_s,
    droplet_radius_cells,
    droplet_radius_cells_from_mass_kg,
    expected_deposited_mass_kg,
    infer_transfer_mode,
    wire_mass_flux_kg_s,
)

if TYPE_CHECKING:
    from ..twin import WAAMTwin

feed_wire = kernels.feed_wire
feed_wire_surface = kernels.feed_wire_surface
feed_wire_momentum = kernels.feed_wire_momentum


def deposition_footprint_cells(twin: "WAAMTwin", drop_r: float | None = None) -> float:
    """Horizontal deposit radius in cells (max of droplet size and arc σ)."""
    drop_r = droplet_radius_cells(twin) if drop_r is None else drop_r
    sigma = twin.sigma_cells * twin.deposition_footprint_sigma_scale
    return max(drop_r, sigma)


__all__ = [
    "feed_wire",
    "feed_wire_surface",
    "feed_wire_momentum",
    "deposition_footprint_cells",
    "droplet_mass_for_interval_kg",
    "wire_mass_flux_kg_s",
    "droplet_mass_kg",
    "droplet_period_s",
    "droplet_radius_cells",
    "droplet_radius_cells_from_mass_kg",
    "expected_deposited_mass_kg",
    "infer_transfer_mode",
]
