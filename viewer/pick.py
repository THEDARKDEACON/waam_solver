"""Screen-center / lookat picking for GGUI probe placement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def lookat_to_grid(
    lookat_mm: tuple[float, float, float],
    twin: "WAAMTwin",
    offset_x_mm: float = 0.0,
) -> tuple[int, int, int]:
    """Map camera lookat (mm) to nearest grid cell index."""
    g = twin.grid
    dx_mm = g.dx * 1000.0
    i = int(np.clip((lookat_mm[0] - offset_x_mm) / dx_mm, 0, g.nx - 1))
    j = int(np.clip(lookat_mm[1] / dx_mm, 0, g.ny - 1))
    k = int(np.clip(lookat_mm[2] / dx_mm, 0, g.nz - 1))
    return i, j, k


def sample_cell(
    twin: "WAAMTwin",
    i: int,
    j: int,
    k: int,
) -> dict[str, float]:
    """Read probe scalars at grid cell (i, j, k)."""
    g = twin.grid
    T = float(g.T[i, j, k])
    return {
        "T_K": T,
        "T_C": T - 273.15,
        "T_max_K": float(g.T_max[i, j, k]),
        "dT_dt_Ks": float(g.dT_dt[i, j, k]),
        "f_l": float(g.f_l[i, j, k]),
        "phi": float(g.phi[i, j, k]),
    }
