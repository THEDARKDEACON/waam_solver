"""Host-side pool velocity and Smagorinsky viscosity diagnostic.

The eddy viscosity is estimated from the resolved strain, nu_t = Cs^2 |S| in
lattice units (dx = 1). The collision kernel uses the non-equilibrium stress
instead, so this ratio is a diagnostic, not that kernel's local value.

Surface-cell liquid fraction and Darcy attenuation say whether the
surface-tension force lands on liquid that is free to move. Attenuation is
``1 + 0.5 K`` with ``K = C (1 - f_l)^2 / f_l^3``, the same factor the
collision kernel divides the velocity by.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_FLOW_KEYS = (
    "u_pool_max_ms",
    "u_surf_p95_ms",
    "nu_ratio",
    "n_pool",
    "surface_fl_median",
    "surface_frac_liquid",
    "surface_atten_median",
)


def _strain_magnitude(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """|S| = sqrt(2 S_ij S_ij) from central differences, dx = 1."""
    dux_dx = np.gradient(ux, axis=0)
    dux_dy = np.gradient(ux, axis=1)
    dux_dz = np.gradient(ux, axis=2)
    duy_dx = np.gradient(uy, axis=0)
    duy_dy = np.gradient(uy, axis=1)
    duy_dz = np.gradient(uy, axis=2)
    duz_dx = np.gradient(uz, axis=0)
    duz_dy = np.gradient(uz, axis=1)
    duz_dz = np.gradient(uz, axis=2)
    sxx = dux_dx
    syy = duy_dy
    szz = duz_dz
    sxy = 0.5 * (dux_dy + duy_dx)
    sxz = 0.5 * (dux_dz + duz_dx)
    syz = 0.5 * (duy_dz + duz_dy)
    s2 = sxx ** 2 + syy ** 2 + szz ** 2 + 2.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    return np.sqrt(np.maximum(2.0 * s2, 0.0))


def _darcy_attenuation(fl: np.ndarray, c_darcy: float) -> np.ndarray:
    """1 + 0.5 K, K = C (1-f_l)^2 / f_l^3. Fully liquid cells stay at 1."""
    fl_c = np.clip(fl.astype(np.float64), 0.0, 1.0)
    solid = np.maximum(1.0 - fl_c, 0.0)
    k_darcy = float(c_darcy) * solid ** 2 / (fl_c ** 3 + 1e-12)
    return 1.0 + 0.5 * k_darcy


def _surface_state(fl: np.ndarray, surface: np.ndarray, c_darcy: float) -> dict[str, float]:
    if not surface.any():
        return {
            "surface_fl_median": 0.0,
            "surface_frac_liquid": 0.0,
            "surface_atten_median": 0.0,
        }
    fl_s = fl[surface].astype(np.float64)
    atten = _darcy_attenuation(fl_s, c_darcy)
    return {
        "surface_fl_median": float(np.median(fl_s)),
        "surface_frac_liquid": float(np.mean(fl_s >= 0.99)),
        "surface_atten_median": float(np.median(atten)),
    }


def pool_flow_stats(twin, cs: float) -> dict[str, float]:
    """One snapshot: pool speed, surface speed, nu_t/nu_mol and surface mush."""
    g = twin.grid
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    ux = g.ux.to_numpy().astype(np.float64)
    uy = g.uy.to_numpy().astype(np.float64)
    uz = g.uz.to_numpy().astype(np.float64)
    pool = ((flags == g.FLAG_FLUID) | (flags == g.FLAG_IFACE)) & (fl >= 0.5)
    gas_above = np.zeros(flags.shape, dtype=bool)
    gas_above[:, :, :-1] = flags[:, :, 1:] == g.FLAG_GAS
    surface = pool & ((flags == g.FLAG_IFACE) | gas_above)
    if hasattr(twin, "effective_C_darcy"):
        c_darcy = float(twin.effective_C_darcy())
    else:
        c_darcy = 0.0
    empty = {key: 0.0 for key in _FLOW_KEYS}
    speed = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    scale = float(g.dx) / float(g.dt)
    n_pool = int(pool.sum())
    if n_pool == 0:
        return empty
    u_pool = float(speed[pool].max()) * scale
    if surface.any():
        u_surf = float(np.percentile(speed[surface], 95)) * scale
    else:
        u_surf = 0.0
    nu_mol = (float(g.tau) - 0.5) / 3.0
    if nu_mol <= 0.0 or float(cs) <= 0.0:
        ratio = 0.0
    else:
        nu_t = float(cs) ** 2 * _strain_magnitude(ux, uy, uz)
        ratio = float(np.mean(nu_t[pool] / nu_mol))
    out = {
        "u_pool_max_ms": u_pool,
        "u_surf_p95_ms": u_surf,
        "nu_ratio": ratio,
        "n_pool": float(n_pool),
    }
    out.update(_surface_state(fl, surface, c_darcy))
    return out


def mean_flow_stats(samples: list[dict[str, float]]) -> dict[str, float]:
    """Mean of snapshot dicts. Empty input returns zeros."""
    if not samples:
        return {key: 0.0 for key in _FLOW_KEYS}
    out: dict[str, Any] = {}
    for key in _FLOW_KEYS:
        out[key] = float(np.mean([row[key] for row in samples]))
    return out
