"""
Surfactant (S/O) effect on surface-tension temperature coefficient.

Mills & Keene (1998): low-S steels → negative dγ/dT (outward Marangoni);
high-S (>60 ppm) → positive dγ/dT (inward, deeper penetration).
"""

from __future__ import annotations

import math


def surfactant_dgamma_dT_scale(sulphur_ppm: float) -> float:
    """
    Return multiplier for base dγ/dT preserving sign convention.

    <30 ppm: outward (negative dγ/dT) — scale magnitude ~1.0
    >60 ppm: inward tendency — flip to positive fraction of |base|
    30–60 ppm: linear blend
    """
    s = max(0.0, sulphur_ppm)
    if s <= 30.0:
        return 1.0
    if s >= 60.0:
        return -0.35
    t = (s - 30.0) / 30.0
    return 1.0 - 1.35 * t


def effective_dgamma_dT(base_dgamma_dT: float, sulphur_ppm: float) -> float:
    """Apply surfactant scaling; base is negative for low-S carbon steel."""
    scale = surfactant_dgamma_dT_scale(sulphur_ppm)
    if scale < 0:
        return -abs(base_dgamma_dT) * scale
    return base_dgamma_dT * scale


def scale_dgamma_table(
    table: list[tuple[float, float]],
    sulphur_ppm: float,
) -> list[tuple[float, float]]:
    """Scale T-dependent dγ/dT table entries."""
    scale = surfactant_dgamma_dT_scale(sulphur_ppm)
    out: list[tuple[float, float]] = []
    for t_k, dg in table:
        if scale < 0:
            out.append((t_k, -abs(dg) * scale))
        else:
            out.append((t_k, dg * scale))
    return out


def lorentz_reference_accel_m_s2(current_A: float, pool_radius_m: float, rho_kg_m3: float) -> float:
    """
    Order-of-magnitude Lorentz acceleration |J×B|/ρ for a GTA pool.

    Uses μ₀ I² / (4π² r² ρ) — Kou/Szekely-type scaling for sanity checks.
    """
    mu0 = 4.0e-7 * math.pi
    r = max(pool_radius_m, 1.0e-4)
    f_vol = mu0 * current_A * current_A / (4.0 * math.pi * math.pi * r * r)
    return f_vol / max(rho_kg_m3, 1.0)
