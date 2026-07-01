"""
test_viewer_extract.py — Viewer Taichi kernels compile and run after init.
"""

from __future__ import annotations

import taichi as ti

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.viewer.extract import (
    FILTER_ALL,
    extract_melt_pool,
    reset_count,
)


def run() -> None:
    init_taichi()
    twin = WAAMTwin(nx=16, ny=12, nz=12, dx=3e-4, max_tracers=10, enable_vof=True)
    twin.reset()
    g = twin.grid

    count = ti.field(dtype=ti.i32, shape=())
    max_cells = g.nx * g.ny * g.nz
    pos = ti.Vector.field(3, dtype=ti.f32, shape=max_cells)
    col = ti.Vector.field(3, dtype=ti.f32, shape=max_cells)

    reset_count(count)
    extract_melt_pool(
        g.f_l, g.T, g.T_max, g.phi, g.flags,
        pos, col, count,
        g.dx * 1000.0, 0.0,
        twin.mat.T_solidus, twin.mat.T_liquidus,
        twin.nz_solid,
        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
        FILTER_ALL, 1, max_cells,
        0, 0, g.ny, g.nz,
    )
    n = int(count[None])
    print(f"[viewer_extract] extract_melt_pool cells={n} arch={ti.lang.impl.current_cfg().arch}")


if __name__ == "__main__":
    run()
    print("PASS")
