"""
test_viewer_streamlines.py — Streamline tracing + GPU buffer upload path.
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from waam_twin.platform import init_taichi
from waam_twin.viewer.streamlines import (
    seeds_in_liquid_near_torch,
    trace_streamlines,
)


def run() -> None:
    init_taichi(backend="cpu")
    nx = ny = nz = 24
    ux = np.zeros((nx, ny, nz), dtype=np.float32)
    uy = np.zeros((nx, ny, nz), dtype=np.float32)
    uz = np.zeros((nx, ny, nz), dtype=np.float32)
    fl = np.zeros((nx, ny, nz), dtype=np.float32)
    flags = np.full((nx, ny, nz), 2, dtype=np.int32)  # gas
    # Circular vortex in liquid slab
    for i in range(6, 18):
        for j in range(6, 18):
            for k in range(4, 12):
                fl[i, j, k] = 1.0
                flags[i, j, k] = 0  # fluid
                cx, cy = 11.5, 11.5
                ux[i, j, k] = -0.02 * (j - cy)
                uy[i, j, k] = 0.02 * (i - cx)

    seeds = seeds_in_liquid_near_torch(
        ux, uy, uz, fl, flags, 12, 12, 8, flag_gas=2, n=12, search_r=6,
    )
    assert len(seeds) >= 4, f"expected liquid seeds, got {seeds}"
    lines = trace_streamlines(
        ux, uy, uz, seeds, n_steps=40, step_cells=0.5,
        f_l=fl, flags=flags, flag_gas=2,
    )
    assert len(lines) >= 2, f"expected streamlines, got {len(lines)}"
    assert all(len(L) >= 3 for L in lines)

    # from_numpy upload path used by the viewer
    sl_vert = ti.Vector.field(3, dtype=ti.f32, shape=4096)
    sl_col = ti.Vector.field(3, dtype=ti.f32, shape=4096)
    vert_np = np.zeros((4096, 3), dtype=np.float32)
    col_np = np.zeros((4096, 3), dtype=np.float32)
    base = 0
    for line in lines:
        for k in range(len(line) - 1):
            vert_np[base] = line[k]
            vert_np[base + 1] = line[k + 1]
            col_np[base] = (0.2, 0.9, 1.0)
            col_np[base + 1] = (0.2, 0.9, 1.0)
            base += 2
    sl_vert.from_numpy(vert_np)
    sl_col.from_numpy(col_np)
    check = sl_vert.to_numpy()
    assert np.linalg.norm(check[0] - vert_np[0]) < 1e-5
    print(f"[streamlines] seeds={len(seeds)}  lines={len(lines)}  segs={base // 2}")


if __name__ == "__main__":
    run()
    print("PASS")
