"""Shared pool- and bead-geometry measurement for calibration and validation."""

from __future__ import annotations

import numpy as np


def measure_pool_mm(twin) -> tuple[float, float, int]:
    """Return (width_mm, depth_mm, n_liquid_cells) from liquid fraction > 0.5.

    Width is the transverse (y) extent at the x-slice through the pool
    centroid; depth is penetration below the substrate top surface. This is
    the standard macrograph W/D definition and matches
    ``WAAMTwin.get_telemetry`` (the previous x/z bounding box measured pool
    LENGTH as width and included the bead crown in depth).
    """
    g = twin.grid
    fl_np = g.f_l.to_numpy()
    liquid_mask = fl_np > 0.5
    n_liq = int(liquid_mask.sum())
    if not np.any(liquid_mask):
        return 0.0, 0.0, 0

    xs = np.nonzero(liquid_mask)[0]
    i_c = int(round(float(xs.mean())))
    sect = liquid_mask[i_c]
    if not sect.any():
        return 0.0, 0.0, n_liq
    y_idx = np.where(sect.any(axis=1))[0]
    z_idx = np.where(sect.any(axis=0))[0]
    W_mm = (y_idx[-1] - y_idx[0] + 1) * g.dx * 1000.0
    nz_solid = int(getattr(twin, "nz_solid", 0))
    D_mm = max(0, nz_solid - z_idx[0]) * g.dx * 1000.0
    return float(W_mm), float(D_mm), n_liq


def measure_bead_frozen_mm(twin) -> dict[str, float]:
    """
    Frozen deposited metal above substrate (mm).

    Uses FLAG_SOLID cells with k >= nz_solid.
    """
    g = twin.grid
    flags = g.flags.to_numpy()
    nz = twin.nz_solid
    solid = flags == g.FLAG_SOLID
    if nz > 0:
        k_ax = np.arange(g.nz)[None, None, :]
        dep = solid & (k_ax >= nz)
    else:
        dep = solid
    if not dep.any():
        return {"bead_width_mm": 0.0, "bead_height_mm": 0.0, "n_deposited_cells": 0.0}

    x_idx = np.where(dep.any(axis=(1, 2)))[0]
    z_idx = np.where(dep.any(axis=(0, 1)))[0]
    h_mm = (z_idx[-1] - nz + 1) * g.dx * 1000.0 if z_idx.size else 0.0
    w_mm = (x_idx[-1] - x_idx[0] + 1) * g.dx * 1000.0 if x_idx.size else 0.0
    return {
        "bead_width_mm": float(max(0.0, w_mm)),
        "bead_height_mm": float(max(0.0, h_mm)),
        "n_deposited_cells": float(dep.sum()),
    }


def measure_bead_metrics(twin) -> dict[str, float]:
    """Pool + frozen bead metrics for validation gates."""
    W, D, n_liq = measure_pool_mm(twin)
    bead = measure_bead_frozen_mm(twin)
    aspect = bead["bead_height_mm"] / max(bead["bead_width_mm"], 0.1)
    pool_aspect = D / max(W, 0.1)
    return {
        "pool_width_mm": W,
        "pool_depth_mm": D,
        "n_liquid_cells": float(n_liq),
        "bead_width_mm": bead["bead_width_mm"],
        "bead_height_mm": bead["bead_height_mm"],
        "bead_aspect_hw": aspect,
        "pool_aspect_dw": pool_aspect,
        "n_deposited_cells": bead["n_deposited_cells"],
    }


def pool_error_pct(W_mm: float, D_mm: float, W_ref: float, D_ref: float) -> float:
    W_err = abs(W_mm - W_ref) / max(W_ref, 0.1) * 100.0
    D_err = abs(D_mm - D_ref) / max(D_ref, 0.1) * 100.0
    return max(W_err, D_err)


def bead_error_pct(metrics: dict[str, float], W_ref: float, D_ref: float, h_ref: float | None = None) -> float:
    err = pool_error_pct(metrics["pool_width_mm"], metrics["pool_depth_mm"], W_ref, D_ref)
    if h_ref is not None and h_ref > 0:
        h_err = abs(metrics["bead_height_mm"] - h_ref) / h_ref * 100.0
        err = max(err, h_err)
    return err
