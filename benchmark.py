"""Shared pool-geometry measurement for calibration and validation."""

from __future__ import annotations

import numpy as np


def measure_pool_mm(twin) -> tuple[float, float, int]:
    """Return (width_mm, depth_mm, n_liquid_cells) from liquid fraction > 0.5."""
    g = twin.grid
    fl_np = g.f_l.to_numpy()
    liquid_mask = fl_np > 0.5
    n_liq = int(liquid_mask.sum())
    if not np.any(liquid_mask):
        return 0.0, 0.0, 0

    x_idx = np.where(liquid_mask.any(axis=(1, 2)))[0]
    z_idx = np.where(liquid_mask.any(axis=(0, 1)))[0]
    W_mm = (x_idx[-1] - x_idx[0] + 1) * g.dx * 1000.0
    D_mm = (z_idx[-1] - z_idx[0] + 1) * g.dx * 1000.0
    return float(W_mm), float(D_mm), n_liq


def pool_error_pct(W_mm: float, D_mm: float, W_ref: float, D_ref: float) -> float:
    W_err = abs(W_mm - W_ref) / max(W_ref, 0.1) * 100.0
    D_err = abs(D_mm - D_ref) / max(D_ref, 0.1) * 100.0
    return max(W_err, D_err)
