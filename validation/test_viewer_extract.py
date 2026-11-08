"""
test_viewer_extract.py — Viewer Taichi kernels compile and run after init.
"""

from __future__ import annotations

import numpy as np

from waam_twin.compiler import ti

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi
from waam_twin.viewer.extract import (
    FILTER_ALL,
    count_nonfinite,
    extract_deposit,
    extract_melt_pool,
    extract_section,
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
        twin.T_amb, twin.mat.T_solidus, twin.mat.T_liquidus,
        twin.nz_solid,
        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
        FILTER_ALL, 1, max_cells,
        0, 0.0, 0.0, 0.0, 0.0,  # no section plane
    )
    n = int(count[None])
    print(f"[viewer_extract] extract_melt_pool cells={n} arch={ti.lang.impl.current_cfg().arch}")
    dx_mm = g.dx * 1000.0
    p = pos.to_numpy()[:n]
    # Particles sit at cell centres, like the φ mesh and the torch markers.
    assert abs(float(p[:, 0].min()) - 0.5 * dx_mm) < 1e-4, p[:, 0].min()

    # Deposit colours: a wire-born cell on top of the plate is copper, plate grey.
    kw = int(twin.nz_solid)
    g.flags[5, 6, kw] = g.FLAG_FLUID
    g.phi[5, 6, kw] = 1.0
    g.alloy_frac[5, 6, kw] = 0.0
    g.f_l[5, 6, kw] = 0.0
    reset_count(count)
    extract_deposit(
        g.alloy_frac, g.f_l, g.T, g.phi, g.flags,
        pos, col, count, dx_mm, 0.0,
        twin.mat.T_solidus, twin.mat.T_liquidus,
        g.FLAG_GAS, g.FLAG_SOLID, FILTER_ALL, 1, max_cells,
        1, 1.0, 0.0, 0.0, 5.0,  # section plane x = 5 cells, keep x ≥ 5
    )
    n = int(count[None])
    p, c = pos.to_numpy()[:n], col.to_numpy()[:n]
    i_idx = np.rint(p[:, 0] / dx_mm - 0.5).astype(int)
    assert i_idx.min() == 5 and i_idx.max() == g.nx - 1, (i_idx.min(), i_idx.max())
    k_idx = np.rint(p[:, 2] / dx_mm - 0.5).astype(int)
    j_idx = np.rint(p[:, 1] / dx_mm - 0.5).astype(int)
    wire = (i_idx == 5) & (j_idx == 6) & (k_idx == kw)
    assert wire.sum() == 1
    cw, cp = c[wire][0], c[~wire].mean(axis=0)
    assert cw[0] - cw[2] > 0.4 and abs(cp[0] - cp[2]) < 0.15, (cw, cp)

    # Section: plane i = 5 holds the plate (φ 1 below the face) and the wire cell.
    sec = ti.field(ti.f32, shape=(max(g.nx, g.ny), g.nz))
    extract_section(g.phi, g.flags, sec, 0, 5, g.ny, g.nz, g.FLAG_GAS)
    a = sec.to_numpy()[: g.ny]
    assert a[6, kw] == 1.0 and a[3, kw] == 0.0 and a[3, kw - 1] == 1.0
    print(f"[viewer_extract] deposit slab cells={n}  section ok")

    # Blow-up detection: max-reductions skip NaN, this count must not.
    assert int(count_nonfinite(g.T, g.H)) == 0
    g.T[3, 3, 3] = float("nan")
    g.H[4, 4, 4] = float("inf")
    assert int(count_nonfinite(g.T, g.H)) == 2
    print("[viewer_extract] non-finite count ok")


if __name__ == "__main__":
    run()
    print("PASS")
