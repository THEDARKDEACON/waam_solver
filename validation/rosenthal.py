"""Analytical Rosenthal solutions for thermal validation."""

from __future__ import annotations

import math


def rosenthal_thick_plate(
    T0: float,
    Q_W: float,
    eta: float,
    k: float,
    rho: float,
    cp: float,
    travel_speed_m_s: float,
    xi_m: float,
    y_m: float = 0.0,
    z_m: float = 0.0,
) -> float:
    """
    3D Rosenthal moving point source on a semi-infinite (thick) plate.

        T - T0 = (ηQ / 2πkR) · exp(−v(R + ξ) / 2α)

    where ξ is the coordinate along travel relative to the source (ξ < 0
    behind the arc), R = √(ξ² + y² + z²) and α = k/(ρcp).
    """
    R = math.sqrt(xi_m * xi_m + y_m * y_m + z_m * z_m)
    if R < 1e-12:
        raise ValueError("Rosenthal solution is singular at the source point")
    alpha = k / (rho * cp)
    return T0 + (eta * Q_W / (2.0 * math.pi * k * R)) * math.exp(
        -travel_speed_m_s * (R + xi_m) / (2.0 * alpha)
    )


def rosenthal_tail_3d(
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
    Trailing-centerline temperature (x < 0 behind the source, y = z = 0).

    On the trailing centerline R = |x| and R + ξ = 0, so the exponential is
    exactly 1 and the tail decays as 1/|x|:

        T - T0 = ηQ / (2πk|x|)

    A previous version multiplied by exp(σx) with σ = ρcp·v/2k — that factor
    belongs to off-axis/leading points, not the trailing centerline, and
    underpredicted the analytical tail (making the CI comparison meaningless).
    """
    if x_tail_m >= 0:
        raise ValueError("x_tail_m must be negative (behind heat source)")
    return rosenthal_thick_plate(
        T0, Q_W, eta, k, rho, cp, travel_speed_m_s, x_tail_m, 0.0, 0.0
    )


# Backwards-compatible alias (the old name mislabelled the solution as 2D and
# used an incorrect exponential decay).
rosenthal_tail_2d = rosenthal_tail_3d
