"""Steady bead cross-section behind the torch.

Width and depth are the liquidus contour of ``T_max`` on each slice (the fusion
envelope). ``H_mm`` is the tallest column of fill, liquid or solid. ``H_frozen_mm``
is the same maximum taken only over columns that have fully solidified, so a
bulging pool tail does not set the bead height. ``area_mm2`` is the deposited
reinforcement area, which the wire feed fixes.

Absolute numbers are always returned. Percentage errors are added only when a
reference with pool_width_mm / pool_depth_mm / bead_height_mm is supplied.
``err_max_pct`` uses width, depth and the frozen height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import _liquidus_K, fusion_contour_mm

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def steady_cross_section(
    twin: "WAAMTwin",
    x0_mm: float,
    x1_mm: float,
    reference: dict[str, Any] | None = None,
    area_expected_mm2: float | None = None,
) -> dict[str, Any]:
    """Mean W/D/H over slices x0_mm..x1_mm, plus frozen height and area."""
    g = twin.grid
    mm_per_cell = g.dx * 1e3
    cell_area = mm_per_cell * mm_per_cell
    i0 = int(round(float(x0_mm) / mm_per_cell))
    i1 = int(round(float(x1_mm) / mm_per_cell))
    i0 = max(0, min(g.nx - 1, i0))
    i1 = max(i0, min(g.nx - 1, i1))

    T_max = g.T_max.to_numpy()
    flags = g.flags.to_numpy()
    metal = flags != g.FLAG_GAS
    thr = float(_liquidus_K(twin))
    nz_solid = int(twin.nz_solid)
    plate_k0 = int(getattr(twin, "plate_k0", 0))

    widths: list[float] = []
    depths: list[float] = []
    for i in range(i0, i1 + 1):
        slab = np.zeros_like(T_max)
        slab[i] = T_max[i]
        out = fusion_contour_mm(
            slab, metal, thr, float(g.dx), nz_solid, 1.0, 0.0, plate_k0,
        )
        widths.append(float(out["fusion_width_mm"]))
        depths.append(float(out["fusion_depth_mm"]))

    fill = np.clip(g.phi.to_numpy(), 0.0, 1.0) * metal
    crown = fill[i0:i1 + 1, :, nz_solid:].sum(axis=2)
    heights = crown.max(axis=1) * mm_per_cell
    deposited = fill[i0:i1 + 1, :, nz_solid:] > 0.5
    n_dep = int(deposited.sum())
    n_frozen_cells = int((flags[i0:i1 + 1, :, nz_solid:] == g.FLAG_SOLID).sum())
    frozen = n_frozen_cells / max(1, n_dep)

    h_frozen: list[float] = []
    areas: list[float] = []
    bead_widths: list[float] = []
    for i in range(i0, i1 + 1):
        col_flags = flags[i, :, nz_solid:]
        col_fill = fill[i, :, nz_solid:]
        is_metal = col_flags != g.FLAG_GAS
        fill_sum = col_fill.sum(axis=1)
        frozen_col = (
            ((col_flags == g.FLAG_SOLID) | ~is_metal).all(axis=1) & is_metal.any(axis=1)
        )
        usable = frozen_col & (fill_sum > 0.0)
        if np.any(usable):
            h_frozen.append(float(fill_sum[usable].max()) * mm_per_cell)
        areas.append(float(col_fill.sum()) * cell_area)
        toes = np.flatnonzero(fill_sum >= 0.5)
        if toes.size:
            bead_widths.append(float(toes[-1] - toes[0] + 1) * mm_per_cell)
        else:
            bead_widths.append(0.0)

    W = float(np.mean(widths))
    D = float(np.mean(depths))
    H = float(np.mean(heights))
    n_slices = int(i1 - i0 + 1)
    n_slices_frozen = len(h_frozen)
    H_frozen = float(np.mean(h_frozen)) if h_frozen else float("nan")
    area = float(np.mean(areas))
    result: dict[str, Any] = {
        "x0_mm": float(x0_mm),
        "x1_mm": float(x1_mm),
        "n_slices": n_slices,
        "n_slices_frozen": n_slices_frozen,
        "W_mm": W,
        "D_mm": D,
        "H_mm": H,
        "H_frozen_mm": H_frozen,
        "W_bead_mm": float(np.mean(bead_widths)),
        "W_sd_mm": float(np.std(widths)),
        "D_sd_mm": float(np.std(depths)),
        "H_sd_mm": float(np.std(heights)),
        "area_WH_mm2": W * H,
        "area_mm2": area,
        "area_sd_mm2": float(np.std(areas)),
        "frozen_frac": float(frozen),
        "steady": bool(frozen >= 0.97 and n_slices_frozen == n_slices),
    }
    if area_expected_mm2 is not None and float(area_expected_mm2) > 0.0:
        expected = float(area_expected_mm2)
        result["area_expected_mm2"] = expected
        result["area_ratio"] = area / expected
    ref = reference or {}
    for key, value, name in (
        ("pool_width_mm", W, "err_W_pct"),
        ("pool_depth_mm", D, "err_D_pct"),
        ("bead_height_mm", H, "err_H_pct"),
        ("bead_height_mm", H_frozen, "err_Hf_pct"),
    ):
        refv = ref.get(key)
        if refv is None or not np.isfinite(value):
            continue
        result[name] = (float(value) / float(refv) - 1.0) * 100.0
    errs = [
        abs(result[k])
        for k in ("err_W_pct", "err_D_pct", "err_Hf_pct")
        if k in result
    ]
    if errs:
        result["err_max_pct"] = max(errs)
    return result
