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


def deposit_rgb(frac: np.ndarray, fl: np.ndarray, temperature: np.ndarray,
                t_solidus: float, t_hi: float) -> np.ndarray:
    """Same map as the particle deposit view: wire copper, plate grey,
    liquid orange → white with temperature."""
    w = np.clip(np.asarray(frac, dtype=np.float32), 0.0, 1.0)[:, None]
    plate = np.array([0.50, 0.53, 0.58], np.float32)
    wire = np.array([0.80, 0.50, 0.22], np.float32)
    c = wire * (1.0 - w) + plate * w
    fl = np.asarray(fl, dtype=np.float32)
    u = np.clip((np.asarray(temperature, np.float32) - t_solidus) / (t_hi - t_solidus + 1e-3), 0.0, 1.0)
    hot = np.stack([np.ones_like(u), 0.45 + 0.5 * u, 0.1 + 0.8 * u], axis=1)
    a = np.where(fl > 0.05, np.minimum(1.0, 0.4 + fl), 0.0)[:, None]
    return (c * (1.0 - a) + hot * a).astype(np.float32)


def phi_isosurface(
    phi: np.ndarray,
    temperature: np.ndarray,
    *,
    dx_mm: float,
    offset_x_mm: float = 0.0,
    nz_solid: int,
    t_lo: float,
    t_hi: float,
    cut: tuple[np.ndarray, np.ndarray] | None = None,
    alloy_frac: np.ndarray | None = None,
    f_l: np.ndarray | None = None,
    t_solidus: float | None = None,
):
    """Triangle sheet at φ = 0.5, coloured by temperature, or by deposit
    (wire vs plate, liquid hot) when ``alloy_frac`` and ``f_l`` are given.

    ``cut`` = (unit normal into the kept side, origin) in world mm, the same
    frame as the returned sheet (x includes ``offset_x_mm``). The sheet is
    clipped there and the cut face is capped with the metal region of that
    plane (φ ≥ 0.5), coloured the same way: a section view.
    None if empty.
    """
    import pyvista as pv

    phi = np.asarray(phi)
    temperature = np.asarray(temperature)
    nx, ny, nz = (int(s) for s in phi.shape)
    k0, k1 = metal_gas_k_band(phi, nz_solid, dx_mm)
    if cut is not None:
        k0 = 0  # a section shows the whole plate thickness
    if k1 <= k0:
        return None
    sl = (slice(0, nx), slice(0, ny), slice(k0, k1))
    slab_phi = np.ascontiguousarray(phi[sl], dtype=np.float32)
    slab_t = np.ascontiguousarray(temperature[sl], dtype=np.float32)
    ni, nj, nk = slab_phi.shape
    grid = pv.ImageData()
    grid.dimensions = (ni + 1, nj + 1, nk + 1)
    grid.spacing = (float(dx_mm), float(dx_mm), float(dx_mm))
    grid.origin = (float(offset_x_mm), 0.0, k0 * float(dx_mm))
    grid.cell_data["phi"] = slab_phi.ravel(order="F")
    grid.cell_data["T"] = slab_t.ravel(order="F")
    deposit = alloy_frac is not None and f_l is not None
    if deposit:
        # φ-weighted so gas cells (born "wire", alloy_frac 0) do not tint the
        # plate surface: frac at a point = avg(frac·φ) / avg(φ).
        fphi = np.asarray(alloy_frac)[sl].astype(np.float32) * np.clip(slab_phi, 0.0, 1.0)
        grid.cell_data["fphi"] = np.ascontiguousarray(fphi).ravel(order="F")
        grid.cell_data["fl"] = np.ascontiguousarray(np.asarray(f_l)[sl], dtype=np.float32).ravel(order="F")
    grid.set_active_scalars("phi")
    points = grid.cell_data_to_point_data()
    try:
        surf = points.contour([0.5], scalars="phi")
    except TypeError:
        surf = points.contour(isosurfaces=[0.5], scalars="phi")
    if surf is None or surf.n_points == 0:
        surf = None

    def _colour(mesh):
        n = mesh.n_points
        t_pts = np.asarray(mesh.point_data.get("T", np.full(n, t_lo)), dtype=np.float32).reshape(-1)
        if t_pts.shape[0] != n:
            t_pts = np.full(n, np.float32(t_lo), dtype=np.float32)
        if deposit:
            phi_arr = mesh.point_data.get("phi")
            phi_pts = np.maximum(
                np.asarray(phi_arr if phi_arr is not None else np.full(n, 0.5), np.float32).reshape(-1),
                1e-3,
            )
            frac = np.asarray(mesh.point_data["fphi"], np.float32).reshape(-1) / phi_pts
            fl_pts = np.asarray(mesh.point_data["fl"], np.float32).reshape(-1)
            t_sol = float(t_solidus) if t_solidus is not None else float(t_lo)
            mesh.point_data["RGB"] = deposit_rgb(frac, fl_pts, t_pts, t_sol, t_hi)
        else:
            mesh.point_data["RGB"] = thermal_rgb(t_pts, t_lo, t_hi)

    if surf is not None:
        _colour(surf)
        surf.compute_normals(
            cell_normals=False,
            point_normals=True,
            consistent_normals=True,
            inplace=True,
            auto_orient_normals=False,
        )
    if cut is None:
        return surf if surf is not None and surf.n_cells > 0 else None

    n_keep = np.asarray(cut[0], dtype=np.float64)
    origin = np.asarray(cut[1], dtype=np.float64)
    parts = []
    if surf is not None:
        clipped = surf.clip(normal=n_keep, origin=origin, invert=False)
        if clipped.n_cells > 0:
            clipped = clipped.extract_surface()
            parts.append(clipped)
    cap = points.slice(normal=n_keep, origin=origin)
    if cap.n_points > 0:
        cap = cap.clip_scalar(scalars="phi", value=0.5, invert=False)
        if cap.n_cells > 0:
            cap = cap.extract_surface()
            _colour(cap)
            # Faces the removed side, toward the viewer.
            cap.point_data["Normals"] = np.tile(-n_keep.astype(np.float32), (cap.n_points, 1))
            parts.append(cap)
    if not parts:
        return None
    keep = ("RGB", "Normals")
    out = None
    for m in parts:
        for name in list(m.point_data.keys()):
            if name not in keep:
                del m.point_data[name]
        for name in list(m.cell_data.keys()):
            del m.cell_data[name]
        out = m if out is None else out.append_polydata(m)
    return out
