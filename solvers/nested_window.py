"""
Phase 5 — nested Tier-F window inside a Tier-C coarse thermal grid.

The coarse twin runs thermal-only (or Tier C). A fine CFD window is placed
around the torch and advanced with full coupled_step; temperatures are
prolonged from coarse→fine at the window boundary and restricted fine→coarse
after each window advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..twin import WAAMTwin


@dataclass
class NestedWindow:
    coarse: "WAAMTwin"
    fine: "WAAMTwin"
    half_width_m: float = 0.012

    def sync_boundary_T(self, torch_x_m: float, torch_y_m: float) -> None:
        """Prolong coarse T into fine halo cells outside the active pool."""
        gc, gf = self.coarse.grid, self.fine.grid
        Tc = gc.T.to_numpy()
        Tf = gf.T.to_numpy()
        # Map fine cell centres → coarse indices
        ox = torch_x_m - self.half_width_m
        oy = torch_y_m - self.half_width_m
        for i in range(gf.nx):
            for j in range(gf.ny):
                for k in range(gf.nz):
                    x = ox + (i + 0.5) * gf.dx
                    y = oy + (j + 0.5) * gf.dx
                    z = (k + 0.5) * gf.dx
                    ic = int(x / gc.dx)
                    jc = int(y / gc.dx)
                    kc = int(z / gc.dx)
                    if 0 <= ic < gc.nx and 0 <= jc < gc.ny and 0 <= kc < gc.nz:
                        # Only overwrite gas / near-boundary fine cells
                        if gf.flags.to_numpy()[i, j, k] == gf.FLAG_GAS:
                            Tf[i, j, k] = Tc[ic, jc, kc]
        gf.T.from_numpy(Tf)

    def restrict_T_to_coarse(self, torch_x_m: float, torch_y_m: float) -> None:
        """Average fine T back onto overlapping coarse cells."""
        gc, gf = self.coarse.grid, self.fine.grid
        Tc = gc.T.to_numpy()
        Tf = gf.T.to_numpy()
        counts = np.zeros_like(Tc)
        ox = torch_x_m - self.half_width_m
        oy = torch_y_m - self.half_width_m
        for i in range(gf.nx):
            for j in range(gf.ny):
                for k in range(gf.nz):
                    x = ox + (i + 0.5) * gf.dx
                    y = oy + (j + 0.5) * gf.dx
                    z = (k + 0.5) * gf.dx
                    ic = int(x / gc.dx)
                    jc = int(y / gc.dx)
                    kc = int(z / gc.dx)
                    if 0 <= ic < gc.nx and 0 <= jc < gc.ny and 0 <= kc < gc.nz:
                        Tc[ic, jc, kc] += Tf[i, j, k]
                        counts[ic, jc, kc] += 1.0
        mask = counts > 0
        Tc[mask] /= counts[mask]
        gc.T.from_numpy(Tc)
