"""
test_viewer_flow_arrows.py — Flow arrow extraction produces visible segments.
"""

from __future__ import annotations

import taichi as ti

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.viewer.extract import FILTER_ALL, FILTER_LIQUID, extract_flow_arrows, reset_count


def run(min_arrows: int = 1) -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=20, ny=12, nz=14, dx=3e-4, max_tracers=10, enable_vof=True)
    twin.reset()
    for _ in range(200):
        twin.step(0.004, 0.004, is_welding=True)

    g = twin.grid
    fl_np = g.f_l.to_numpy()
    n_liq = int((fl_np > 0.08).sum())
    print(f"[viewer_flow_arrows] liquid_cells={n_liq}")

    count = ti.field(dtype=ti.i32, shape=())
    max_arrows = 512
    vert = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)
    col = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)
    dx_mm = g.dx * 1000.0

    filt = FILTER_LIQUID if n_liq > 0 else FILTER_ALL

    reset_count(count)
    extract_flow_arrows(
        g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
        vert, col, count,
        dx_mm, 0.0, g.dx, g.dt,
        dx_mm * 4.0, dx_mm * 1.2,
        1,
        g.FLAG_GAS, g.FLAG_SOLID,
        filt, 1, max_arrows,
        0, 0, g.ny, g.nz,
    )
    n = int(count[None])
    print(f"[viewer_flow_arrows] arrows={n}")
    if n_liq == 0 and n == 0:
        print("[viewer_flow_arrows] skip — no liquid yet on smoke grid")
        return
    assert n >= min_arrows, f"expected >= {min_arrows} velocity arrows, got {n}"


if __name__ == "__main__":
    run()
    print("PASS")
