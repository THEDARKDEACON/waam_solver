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
    pmat = getattr(twin, "plate_mat", None)
    if pmat is not None:
        return float(pmat.T_liquidus)
    return float(twin.mat.T_liquidus)


def _interp_crossing(a: float, b: float, level: float) -> float:
    """Fractional position in [0, 1] where a→b crosses level."""
    den = b - a
    if abs(den) < 1e-30:
        return 0.5
    t = (level - a) / den
    return float(np.clip(t, 0.0, 1.0))


def _finite_max(values: np.ndarray, axis: int) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, -np.inf)
    reduced = filled.max(axis=axis)
    return np.where(reduced > -1e20, reduced, np.nan)


def _profile_edges(profile: np.ndarray, thr: float) -> tuple[float, float] | None:
    """Crossings in cell-center coordinates. Samples live at index + 0.5."""
    finite = np.isfinite(profile)
    above = np.flatnonzero(finite & (profile >= thr))
    if above.size == 0:
        return None
    i0 = int(above[0])
    i1 = int(above[-1])
    if i0 > 0 and finite[i0 - 1] and profile[i0 - 1] < thr:
        denom = float(profile[i0] - profile[i0 - 1])
        t = (thr - float(profile[i0 - 1])) / denom if abs(denom) > 1e-12 else 0.5
        left = (i0 - 1 + 0.5) + float(np.clip(t, 0.0, 1.0))
    else:
        left = float(i0)
    if i1 + 1 < profile.size and finite[i1 + 1] and profile[i1 + 1] < thr:
        denom = float(profile[i1 + 1] - profile[i1])
        t = (thr - float(profile[i1])) / denom if abs(denom) > 1e-12 else 0.5
        right = (i1 + 0.5) + float(np.clip(t, 0.0, 1.0))
    else:
        right = float(i1 + 1)
    return left, right


def _depth_mm(col: np.ndarray, thr: float, nz_solid: int, dx_m: float,
              plate_k0: int = 0) -> float:
    """Depth from the plate face (index nz_solid) to the threshold crossing."""
    if nz_solid <= 0:
        return 0.0
    finite = np.isfinite(col)
    kk = np.arange(col.size)
    above = np.flatnonzero(
        finite & (col >= thr) & (kk >= plate_k0) & (kk < nz_solid)
    )
    if above.size == 0:
        return 0.0
    k_bot = int(above[0])
    if k_bot > 0 and finite[k_bot - 1] and col[k_bot - 1] < thr:
        denom = float(col[k_bot] - col[k_bot - 1])
        t = (thr - float(col[k_bot - 1])) / denom if abs(denom) > 1e-12 else 0.5
        k_cross = (k_bot - 1 + 0.5) + float(np.clip(t, 0.0, 1.0))
    else:
        k_cross = float(k_bot)
    return float(max(0.0, (nz_solid - k_cross) * dx_m * 1000.0))


def fusion_contour_mm(
    T_max: np.ndarray,
    metal: np.ndarray,
    thr: float,
    dx_m: float,
    nz_solid: int,
    dir_x: float,
    dir_y: float,
    plate_k0: int = 0,
) -> dict[str, float]:
    """1520-style contour. Width is perpendicular to travel; depth uses cell centers.

    Temperatures are sampled at cell centers (index + 0.5). The plate face is
    the index ``nz_solid``. Width is the distance between the two threshold
    crossings on the travel station that holds the hottest metal.
    """
    empty = {
        "pool_width_mm": 0.0,
        "pool_depth_mm": 0.0,
        "pool_width_fusion_mm": 0.0,
        "pool_depth_fusion_mm": 0.0,
        "fusion_width_mm": 0.0,
        "fusion_depth_mm": 0.0,
        "n_fusion_cells": 0.0,
    }
    if T_max.size == 0 or nz_solid <= 0 or not np.any(metal):
        return empty
    n_fused = float(np.sum(metal & (T_max >= thr)))
    work = np.where(metal, T_max, np.nan)
    if not np.isfinite(work).any() or float(np.nanmax(work)) < thr:
        empty["n_fusion_cells"] = n_fused
        return empty
    peak = np.unravel_index(int(np.nanargmax(work)), work.shape)
    travel_y = abs(float(dir_y)) > abs(float(dir_x))
    if travel_y:
        slab = work[:, int(peak[1]), :]
    else:
        slab = work[int(peak[0]), :, :]
    prof = _finite_max(slab, axis=1)
    if not np.isfinite(prof).any():
        empty["n_fusion_cells"] = n_fused
        return empty
    line = int(np.nanargmax(prof))
    col = slab[line, :]
    edges = _profile_edges(prof, thr)
    width = 0.0 if edges is None else max(0.0, (edges[1] - edges[0]) * dx_m * 1000.0)
    depth = _depth_mm(col, thr, nz_solid, dx_m, plate_k0)
    return {
        "pool_width_mm": float(width),
        "pool_depth_mm": float(depth),
        "pool_width_fusion_mm": float(width),
        "pool_depth_fusion_mm": float(depth),
        "fusion_width_mm": float(width),
        "fusion_depth_mm": float(depth),
        "n_fusion_cells": n_fused,
    }


def measure_fusion_zone_mm(twin: "WAAMTwin", T_threshold_K: float | None = None) -> dict[str, float]:
    """Fusion width perpendicular to travel, depth below the plate face.

    Both edges are interpolated between cell centers on the maximum-temperature
    field. The default threshold is the plate liquidus when a plate card is loaded.
    """
    g = twin.grid
    T_max = g.T_max.to_numpy()
    flags = g.flags.to_numpy()
    thr = float(_liquidus_K(twin) if T_threshold_K is None else T_threshold_K)
    dx, dy, _dz = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
    metal = flags != g.FLAG_GAS
    out = fusion_contour_mm(
        T_max, metal, thr, float(g.dx), int(getattr(twin, "nz_solid", 0)),
        float(dx), float(dy),
        int(getattr(twin, "plate_k0", 0)),
    )
    out["T_threshold_K"] = thr
    return out


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
