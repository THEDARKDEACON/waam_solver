"""Upload material property tables to Taichi fields for GPU lookup."""

from __future__ import annotations

import numpy as np
import taichi as ti

from .materials import MaterialProps

MAX_KNOTS = 8


class MaterialGPUTables:
    """Piecewise-linear material knots on GPU (max 8 points per property)."""

    def __init__(self, mat: MaterialProps):
        tbl = mat.tables
        self.enabled = bool(tbl.cp or tbl.k or tbl.mu or tbl.dgamma_dT)
        self.cp_T = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.cp_V = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.k_T = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.k_V = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.mu_T = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.mu_V = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.dgamma_T = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.dgamma_V = ti.field(dtype=ti.f32, shape=(MAX_KNOTS,))
        self.n_cp = ti.field(dtype=ti.i32, shape=())
        self.n_k = ti.field(dtype=ti.i32, shape=())
        self.n_mu = ti.field(dtype=ti.i32, shape=())
        self.n_dgamma = ti.field(dtype=ti.i32, shape=())
        self.cp_fallback = float(mat.cp)
        self.k_fallback = float(mat.k)
        self.mu_fallback = float(mat.mu)
        self.dgamma_fallback = float(mat.dgamma_dT)
        self._upload(mat)

    def _upload_knots(
        self,
        knots: list[tuple[float, float]],
        t_field: ti.Field,
        v_field: ti.Field,
        n_field: ti.Field,
        fallback: float,
    ) -> None:
        n = min(len(knots), MAX_KNOTS)
        t_arr = np.zeros(MAX_KNOTS, dtype=np.float32)
        v_arr = np.zeros(MAX_KNOTS, dtype=np.float32)
        if n == 0:
            t_arr[0] = 300.0
            v_arr[0] = fallback
            n = 1
        else:
            for i in range(n):
                t_arr[i] = knots[i][0]
                v_arr[i] = knots[i][1]
        t_field.from_numpy(t_arr)
        v_field.from_numpy(v_arr)
        n_field[None] = n

    def _upload(self, mat: MaterialProps) -> None:
        tbl = mat.tables
        self._upload_knots(tbl.cp, self.cp_T, self.cp_V, self.n_cp, mat.cp)
        self._upload_knots(tbl.k, self.k_T, self.k_V, self.n_k, mat.k)
        self._upload_knots(tbl.mu, self.mu_T, self.mu_V, self.n_mu, mat.mu)
        self._upload_knots(tbl.dgamma_dT, self.dgamma_T, self.dgamma_V, self.n_dgamma, mat.dgamma_dT)
