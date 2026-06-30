"""Analytical Rosenthal solutions for thermal validation."""

from __future__ import annotations

import math


def rosenthal_tail_2d(
    T0: float,
    Q_W: float,
    eta: float,
    k: float,
    rho: float,
    cp: float,
    travel_speed_m_s: float,
    x_tail_m: float,
) -> float:
    """
    2D Rosenthal trailing-edge temperature (x < 0 behind a point source at x=0).

    T - T0 = (ηQ / 2πk) · (1/|x|) · exp(σx)   where σ = ρcpv / 2k
    """
    if x_tail_m >= 0:
        raise ValueError("x_tail_m must be negative (behind heat source)")
    sigma = rho * cp * travel_speed_m_s / (2.0 * k)
    return T0 + (eta * Q_W / (2.0 * math.pi * k)) * (1.0 / abs(x_tail_m)) * math.exp(sigma * x_tail_m)
