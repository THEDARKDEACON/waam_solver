"""
Sub-cell bead / fusion-zone geometry.

Primary fusion metric: liquidus isosurface of T_max (fusion envelope).
Secondary: liquid-fraction bounding box (legacy).
Bead crown: free-surface (phi / f_l) isosurface above the plate.
Toe angle: surface normal at the triple line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def _liquidus_K(twin: "WAAMTwin") -> float:
    return float(twin.mat.T_liquidus)


def _interp_crossing(a: float, b: float, level: float) -> float:
    """Fractional position in [0, 1] where a→b crosses level."""
    den = b - a
    if abs(den) < 1e-30:
        return 0.5
    t = (level - a) / den
    return float(np.clip(t, 0.0, 1.0))


def measure_fusion_zone_mm(twin: "WAAMTwin") -> dict[str, float]:
    """
    Fusion W/D from liquidus isosurface of T_max with sub-cell interpolation.

    Width: transverse (y) extent at the x-slice through the fusion centroid.
    Depth: penetration below the substrate top (nz_solid).
    """
    g = twin.grid
    T_max = g.T_max.to_numpy()
    flags = g.flags.to_numpy()
    T_liq = _liquidus_K(twin)
    metal = flags != g.FLAG_GAS
    fused = metal & (T_max >= T_liq)
    n_fused = int(fused.sum())
    if n_fused == 0:
        return {
            "pool_width_mm": 0.0,
            "pool_depth_mm": 0.0,
            "pool_width_fusion_mm": 0.0,
            "pool_depth_fusion_mm": 0.0,
            "fusion_width_mm": 0.0,
            "fusion_depth_mm": 0.0,
            "n_fusion_cells": 0.0,
        }

    xs = np.nonzero(fused)[0]
    i_c = int(round(float(xs.mean())))
    nz = int(getattr(twin, "nz_solid", 0))
    sect = T_max[i_c]  # (ny, nz)
    metal_s = metal[i_c]

    # Width: scan y at each k near surface; find outer liquidus crossings
    y_lo = None
    y_hi = None
    for j in range(g.ny - 1):
        for k in range(g.nz):
            if not (metal_s[j, k] or metal_s[j + 1, k]):
                continue
            a, b = sect[j, k], sect[j + 1, k]
            if (a >= T_liq) != (b >= T_liq) or (a >= T_liq and b >= T_liq):
                if a >= T_liq or b >= T_liq:
                    if y_lo is None:
                        if a < T_liq <= b:
                            y_lo = j + _interp_crossing(a, b, T_liq)
                        elif a >= T_liq:
                            y_lo = float(j)
                    if b < T_liq <= a or b >= T_liq:
                        if b < T_liq <= a:
                            y_hi = j + _interp_crossing(a, b, T_liq)
                        else:
                            y_hi = float(j + 1)

    # Robust bbox fallback with half-cell correction
    y_idx = np.where(fused[i_c].any(axis=1))[0]
    z_idx = np.where(fused[i_c].any(axis=0))[0]
    if y_idx.size == 0:
        return {
            "pool_width_mm": 0.0,
            "pool_depth_mm": 0.0,
            "pool_width_fusion_mm": 0.0,
            "pool_depth_fusion_mm": 0.0,
            "fusion_width_mm": 0.0,
            "fusion_depth_mm": 0.0,
            "n_fusion_cells": float(n_fused),
        }

    # Refine left/right edges on the mid-k row of the fused band
    k_mid = int(z_idx.mean())
    row = sect[:, k_mid]
    j0, j1 = int(y_idx[0]), int(y_idx[-1])
    # left edge
    if j0 > 0 and row[j0 - 1] < T_liq <= row[j0]:
        y_left = (j0 - 1) + _interp_crossing(row[j0 - 1], row[j0], T_liq)
    else:
        y_left = float(j0) - 0.5
    # right edge
    if j1 + 1 < g.ny and row[j1] >= T_liq > row[j1 + 1]:
        y_right = j1 + _interp_crossing(row[j1], row[j1 + 1], T_liq)
    else:
        y_right = float(j1) + 0.5

    W_mm = max(0.0, (y_right - y_left) * g.dx * 1000.0)

    # Depth: bottom liquidus crossing below nz_solid
    col = sect[int(round(0.5 * (y_left + y_right))), :]
    k_bot = int(z_idx[0])
    if k_bot > 0 and col[k_bot - 1] < T_liq <= col[k_bot]:
        k_cross = (k_bot - 1) + _interp_crossing(col[k_bot - 1], col[k_bot], T_liq)
    else:
        k_cross = float(k_bot) - 0.5
    D_mm = max(0.0, (nz - k_cross) * g.dx * 1000.0) if nz > 0 else 0.0

    return {
        "pool_width_mm": float(W_mm),
        "pool_depth_mm": float(D_mm),
        "pool_width_fusion_mm": float(W_mm),
        "pool_depth_fusion_mm": float(D_mm),
        # Aliases for benchmark.py / mesh_convergence_report callers
        "fusion_width_mm": float(W_mm),
        "fusion_depth_mm": float(D_mm),
        "n_fusion_cells": float(n_fused),
    }


def measure_bead_crown_mm(twin: "WAAMTwin") -> dict[str, float]:
    """Bead width/height from phi isosurface above the plate (sub-cell)."""
    g = twin.grid
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    nz = int(getattr(twin, "nz_solid", 0))
    metal = (flags != g.FLAG_GAS) & (phi > 0.5)
    if nz > 0:
        metal[:, :, :nz] = False
    if not metal.any():
        return {"bead_width_mm": 0.0, "bead_height_mm": 0.0, "n_crown_cells": 0.0}

    y_idx = np.where(metal.any(axis=(0, 2)))[0]
    z_idx = np.where(metal.any(axis=(0, 1)))[0]
    w_mm = (y_idx[-1] - y_idx[0] + 1) * g.dx * 1000.0 if y_idx.size else 0.0
    # Height: top free-surface crossing above plate
    if z_idx.size:
        # average top k over deposit footprint
        tops = []
        for i in range(g.nx):
            for j in range(g.ny):
                col = metal[i, j, :]
                if col.any():
                    tops.append(int(np.where(col)[0][-1]))
        if tops:
            k_top = float(np.mean(tops)) + 0.5
            h_mm = max(0.0, (k_top - nz + 1) * g.dx * 1000.0)
        else:
            h_mm = 0.0
    else:
        h_mm = 0.0
    return {
        "bead_width_mm": float(max(0.0, w_mm)),
        "bead_height_mm": float(max(0.0, h_mm)),
        "n_crown_cells": float(metal.sum()),
    }


def measure_toe_angle_deg(twin: "WAAMTwin") -> float:
    """Contact / toe angle [deg] from phi gradient at substrate triple line."""
    g = twin.grid
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    nz = int(getattr(twin, "nz_solid", 0))
    if nz < 1 or nz >= g.nz - 1:
        return float("nan")
    # cells just above plate with interface
    k = nz
    iface = (flags[:, :, k] != g.FLAG_GAS) & (phi[:, :, k] > 0.2) & (phi[:, :, k] < 0.9)
    if not iface.any():
        return float("nan")
    # approximate n from central differences in yz at triple line
    angles = []
    for i, j in zip(*np.where(iface)):
        dpy = 0.0
        if 0 < j < g.ny - 1:
            dpy = 0.5 * (phi[i, j + 1, k] - phi[i, j - 1, k])
        dpz = phi[i, j, min(k + 1, g.nz - 1)] - phi[i, j, k]
        nmag = np.hypot(dpy, dpz)
        if nmag < 1e-8:
            continue
        # angle of interface normal to wall normal (+z)
        cos_a = abs(dpz) / nmag
        angles.append(float(np.degrees(np.arccos(np.clip(cos_a, 0.0, 1.0)))))
    if not angles:
        return float("nan")
    return float(np.median(angles))


def measure_pool_length_mm(twin: "WAAMTwin") -> float:
    """Travel-direction (x) extent of liquid cells [mm]."""
    g = twin.grid
    fl = g.f_l.to_numpy() > 0.5
    if not fl.any():
        # fall back to fusion envelope
        T_max = g.T_max.to_numpy()
        flags = g.flags.to_numpy()
        fused = (flags != g.FLAG_GAS) & (T_max >= _liquidus_K(twin))
        if not fused.any():
            return 0.0
        xs = np.where(fused.any(axis=(1, 2)))[0]
    else:
        xs = np.where(fl.any(axis=(1, 2)))[0]
    if xs.size == 0:
        return 0.0
    return float((xs[-1] - xs[0] + 1) * g.dx * 1000.0)


def measure_all(twin: "WAAMTwin") -> dict[str, Any]:
    fz = measure_fusion_zone_mm(twin)
    crown = measure_bead_crown_mm(twin)
    return {
        **fz,
        **crown,
        "toe_angle_deg": measure_toe_angle_deg(twin),
        "pool_length_mm": measure_pool_length_mm(twin),
    }
