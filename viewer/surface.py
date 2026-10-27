"""φ = 0.5 metal/gas sheet for the live viewer.

Marching cubes on the volume fraction, cropped to the band where metal
meets gas: a few millimetres under the plate face through the bead crown.
The plate interior and the empty gas above the crown are left out.
"""

from __future__ import annotations

import numpy as np

# Pool depth on the shop lock is ~2 mm. Six millimetres under the face
# keeps a deeper pool on the sheet without walking the whole coupon.
_BELOW_FACE_MM = 6.0
_ABOVE_CROWN_CELLS = 2


def metal_gas_k_band(
    phi: np.ndarray,
    nz_solid: int,
    dx_mm: float,
    *,
    below_mm: float = _BELOW_FACE_MM,
) -> tuple[int, int]:
    """Cell range [k0, k1) that contains the metal/gas crossing.

    k1 is exclusive. The range always includes one gas layer above the
    highest metal cell so the crown is closed, and `below_mm` of metal
    under the plate face so a depressed pool stays on the sheet.
    """
    nz = int(phi.shape[2])
    k_face = int(np.clip(nz_solid, 0, nz))
    n_below = max(2, int(np.ceil(float(below_mm) / max(float(dx_mm), 1e-6))))
    k0 = max(0, k_face - n_below)
    occupied = np.where((phi > 0.05).any(axis=(0, 1)))[0]
    k_top = int(occupied[-1]) if occupied.size else max(0, k_face - 1)
    k1 = min(nz, max(k_top + 1 + _ABOVE_CROWN_CELLS, k_face + 1))
    if k1 <= k0:
        k1 = min(nz, k0 + 1)
    return k0, k1


def _clip_span(n: int, enabled: bool) -> tuple[int, int]:
    """Mid-plane slab of 3 cells, matching the particle clip (±1)."""
    if not enabled:
        return 0, n
    mid = n // 2
    return max(0, mid - 1), min(n, mid + 2)


def thermal_rgb(temperature: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """Same blue → white map as the temperature particle view."""
    u = np.clip(
        (np.asarray(temperature, dtype=np.float32) - np.float32(t_lo))
        / (np.float32(t_hi) - np.float32(t_lo) + np.float32(1e-3)),
        0.0,
        1.0,
    )
    c = np.empty(u.shape + (3,), dtype=np.float32)
    c[:] = (0.05, 0.08, 0.35)
    m0 = u < 0.25
    t = u / np.float32(0.25)
    c[m0, 0] = 0.05
    c[m0, 1] = 0.08 + 0.55 * t[m0]
    c[m0, 2] = 0.35 + 0.55 * t[m0]
    m1 = (u >= 0.25) & (u < 0.5)
    t = (u - np.float32(0.25)) / np.float32(0.25)
    c[m1, 0] = 0.1 + 0.85 * t[m1]
    c[m1, 1] = 0.63 + 0.27 * t[m1]
    c[m1, 2] = 0.9 - 0.7 * t[m1]
    m2 = (u >= 0.5) & (u < 0.75)
    t = (u - np.float32(0.5)) / np.float32(0.25)
    c[m2, 0] = 0.95
    c[m2, 1] = 0.9 - 0.55 * t[m2]
    c[m2, 2] = 0.2 * (1.0 - t[m2])
    m3 = u >= 0.75
    t = (u - np.float32(0.75)) / np.float32(0.25)
    c[m3, 0] = 0.95 + 0.05 * t[m3]
    c[m3, 1] = 0.35 + 0.55 * t[m3]
    c[m3, 2] = 0.15 + 0.7 * t[m3]
    return c


def phi_isosurface(
    phi: np.ndarray,
    temperature: np.ndarray,
    *,
    dx_mm: float,
    offset_x_mm: float = 0.0,
    nz_solid: int,
    t_lo: float,
    t_hi: float,
    clip_x: bool = False,
    clip_y: bool = False,
    clip_z: bool = False,
):
    """Triangle sheet at φ = 0.5, colored by temperature. None if empty."""
    import pyvista as pv

    phi = np.asarray(phi)
    temperature = np.asarray(temperature)
    nx, ny, nz = (int(s) for s in phi.shape)
    if clip_z:
        k0, k1 = _clip_span(nz, True)
    else:
        k0, k1 = metal_gas_k_band(phi, nz_solid, dx_mm)
    i0, i1 = _clip_span(nx, clip_x)
    j0, j1 = _clip_span(ny, clip_y)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return None

    slab_phi = np.ascontiguousarray(phi[i0:i1, j0:j1, k0:k1], dtype=np.float32)
    slab_t = np.ascontiguousarray(temperature[i0:i1, j0:j1, k0:k1], dtype=np.float32)
    ni, nj, nk = slab_phi.shape
    grid = pv.ImageData()
    grid.dimensions = (ni + 1, nj + 1, nk + 1)
    grid.spacing = (float(dx_mm), float(dx_mm), float(dx_mm))
    grid.origin = (
        float(offset_x_mm) + i0 * float(dx_mm),
        j0 * float(dx_mm),
        k0 * float(dx_mm),
    )
    grid.cell_data["phi"] = slab_phi.ravel(order="F")
    grid.cell_data["T"] = slab_t.ravel(order="F")
    grid.set_active_scalars("phi")
    points = grid.cell_data_to_point_data()
    try:
        surf = points.contour([0.5], scalars="phi")
    except TypeError:
        surf = points.contour(isosurfaces=[0.5], scalars="phi")
    if surf is None or surf.n_cells == 0 or surf.n_points == 0:
        return None

    t_pts = np.asarray(surf.point_data.get("T", t_lo), dtype=np.float32).reshape(-1)
    if t_pts.shape[0] != surf.n_points:
        t_pts = np.full(surf.n_points, np.float32(t_lo), dtype=np.float32)
    surf.point_data["RGB"] = thermal_rgb(t_pts, t_lo, t_hi)
    surf.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        inplace=True,
        auto_orient_normals=False,
    )
    return surf
