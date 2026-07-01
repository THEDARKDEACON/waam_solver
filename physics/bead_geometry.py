"""Bead geometry telemetry helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid


def estimate_toe_angle_deg(twin: "WAAMTwin", g: "WAAMGrid") -> float:
    """
    Estimate contact-line slope at substrate from φ field (centre column).

    Returns angle in degrees through the liquid phase; 90° = vertical wetting.
    """
    import numpy as np

    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    j = g.ny // 2
    i = g.nx // 3
    nz_ref = twin.nz_solid
    k0 = max(0, nz_ref - 1)
    k1 = min(g.nz - 1, nz_ref + 8)
    best_angle = float(twin.contact_angle_deg)
    best_phi = 0.0
    for k in range(k0, k1):
        if flags[i, j, k] == g.FLAG_GAS and phi[i, j, k] > best_phi:
            best_phi = phi[i, j, k]
        if phi[i, j, k] > 0.45 and k > k0:
            dz = g.dx
            dphi = phi[i, j, k] - phi[i, j, k - 1]
            if abs(dphi) > 1e-4:
                slope = abs(dphi) * g.dx / max(dz, 1e-9)
                angle = math.degrees(math.atan(1.0 / max(slope, 1e-6)))
                best_angle = min(89.0, max(10.0, angle))
    return best_angle


def bead_reinforcement_height_mm(twin: "WAAMTwin", g: "WAAMGrid") -> float:
    """Solidified metal height above substrate (mm)."""
    from .electrical_stickout import measure_bead_height_m
    return measure_bead_height_m(twin, g) * 1000.0
