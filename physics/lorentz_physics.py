"""Lorentz (J×B) reference scales in SI units."""

from __future__ import annotations

import math

from .surfactant import lorentz_reference_accel_m_s2

__all__ = ["lorentz_reference_accel_m_s2", "lorentz_body_force_peak_N_m3"]


def lorentz_body_force_peak_N_m3(current_A: float, pool_radius_m: float) -> float:
    """
    Peak |J×B| body-force scale [N/m³].

    J ~ I/(π r²), B ~ μ0 I/(2π r) → |J×B| ~ μ0 I² / (2 π² r³).
    (Earlier revision used I²/r², which is a magnetic-pressure scale [N/m²].)
    """
    mu0 = 4.0e-7 * math.pi
    r = max(pool_radius_m, 1.0e-4)
    return mu0 * current_A * current_A / (2.0 * math.pi * math.pi * r ** 3)
