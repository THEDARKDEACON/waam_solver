"""Bead size on one cross-section behind the torch (viewer HUD).

The section is a transverse plane where the torch was ``lag`` mm ago, so it
cuts the bead the arc has just laid down rather than the moving pool. On
that plane every column gets its metal height above the plate face,
h(t) = dx · Σ φ (sub-cell, liquid included, all layers of a wall). The bead
is the run of columns joined to the path centreline whose height exceeds a
threshold; spatter or a neighbouring bead that is not connected to it is
ignored.

  height = max h over the run
  width  = distance between the threshold crossings at the two edges
  toe    = mean over both edges of atan(steepest slope of h within 3 columns
           of the edge): a coarse wetting-angle reading, about one cell wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BeadSection:
    height_mm: float
    width_mm: float
    toe_deg: float
    station_mm: float  # path coordinate (x or y, world mm) of the plane
    axis: int  # 0: plane i = const (travel along x); 1: plane j = const


def column_heights_mm(section: np.ndarray, k_face: int, dx_mm: float) -> np.ndarray:
    """Metal height above the plate face per column (section is [t, k] of φ)."""
    sec = np.asarray(section, dtype=np.float64)
    k_face = int(np.clip(k_face, 0, sec.shape[1]))
    return sec[:, k_face:].sum(axis=1) * float(dx_mm)


def _crossing(h_out: float, h_in: float, thr: float) -> float:
    """Fraction from the outside column centre to the threshold crossing."""
    d = h_in - h_out
    if d <= 1e-12:
        return 1.0
    return float(np.clip((thr - h_out) / d, 0.0, 1.0))


def measure(section: np.ndarray, k_face: int, t_centre: int, dx_mm: float) -> tuple[float, float, float]:
    """(height_mm, width_mm, toe_deg) of the bead through column t_centre.

    Zeros when no bead column lies within 3 cells of the centreline.
    """
    h = column_heights_mm(section, k_face, dx_mm)
    n = h.size
    if n == 0:
        return 0.0, 0.0, 0.0
    hmax_all = float(h.max())
    thr = max(0.25 * dx_mm, 0.05 * hmax_all)
    above = h >= thr
    tc = int(np.clip(t_centre, 0, n - 1))
    if not above[tc]:
        cands = [t for t in range(max(0, tc - 3), min(n, tc + 4)) if above[t]]
        if not cands:
            return 0.0, 0.0, 0.0
        tc = min(cands, key=lambda t: abs(t - tc))
    t0 = tc
    while t0 > 0 and above[t0 - 1]:
        t0 -= 1
    t1 = tc
    while t1 < n - 1 and above[t1 + 1]:
        t1 += 1
    run = h[t0:t1 + 1]
    height = float(run.max())
    left = t0 - 1 + _crossing(h[t0 - 1], h[t0], thr) if t0 > 0 else -0.5
    right = t1 + 1 - _crossing(h[t1 + 1], h[t1], thr) if t1 < n - 1 else n - 0.5
    width = max(0.0, (right - left) * dx_mm)

    toes = []
    if t0 > 0:  # slopes h[t+1]-h[t] from the outside column inward, inside the run
        s = max(h[t + 1] - h[t] for t in range(t0 - 1, min(t0 + 2, t1)))
        toes.append(math.degrees(math.atan(max(float(s), 0.0) / dx_mm)))
    if t1 < n - 1:
        s = max(h[t - 1] - h[t] for t in range(t1 + 1, max(t1 - 2, t0), -1))
        toes.append(math.degrees(math.atan(max(float(s), 0.0) / dx_mm)))
    toe = float(np.mean(toes)) if toes else 0.0
    return height, width, toe
