"""
test_viewer_cut.py — viewer section plane and probe ray picking.

  - CutPlane: torch-riding planes start behind the arc across the travel
    direction; nudging, torch ↔ fixed anchor switching without a jump; the
    kept side faces away from the camera; flip; free (tilted) plane.
  - ray_pick_metal: a ray down onto a plate finds its face; with a cut that
    removes the near half, a ray hits the cut face instead of the removed
    metal in front of it.
"""

from __future__ import annotations

import numpy as np

from waam_twin.compiler import ti
from waam_twin.viewer.cut import FREE, CutPlane


def _cut_plane() -> None:
    torch = np.array([30.0, 20.0, 9.0])
    face = 8.8
    cam = np.array([10.0, -10.0, 30.0])
    c = CutPlane()
    assert c.plane(torch, face, cam) is None
    c.select(0, (1.0, 0.0, 0.0), behind_mm=4.0)  # travel +x → 4 mm behind the arc
    n, o = c.plane(torch, face, cam)
    assert abs(o[0] - 26.0) < 1e-9, o
    assert n[0] == 1.0, n  # camera at x=10 < 26: keep +x so the cut face looks at it
    assert "behind arc" in c.label(torch, face, (1.0, 0.0, 0.0))
    c.nudge(-1.0)
    assert abs(c.position_mm(torch, face) - 25.0) < 1e-9
    moved = torch + np.array([5.0, 0.0, 0.0])
    assert abs(c.position_mm(moved, face) - 30.0) < 1e-9  # rides with the torch
    c.toggle_anchor(torch, face)
    assert c.anchor == "fixed" and abs(c.position_mm(moved, face) - 25.0) < 1e-9
    c.toggle_anchor(moved, face)  # back to torch: no jump
    assert c.anchor == "torch" and abs(c.position_mm(moved, face) - 25.0) < 1e-9
    c.toggle_flip()
    n2, _ = c.plane(torch, face, cam)
    assert n2[0] == -1.0
    c.select(0, (1.0, 0.0, 0.0))
    assert not c.active  # same key again: off

    c.select(1, (1.0, 0.0, 0.0))  # along the bead: through the torch centreline
    _, o = c.plane(torch, face, cam)
    assert abs(o[1] - 20.0) < 1e-9
    c.select(2, (1.0, 0.0, 0.0), z_offset_mm=-0.2)  # plan view, top plate layer
    n, o = c.plane(torch, face, cam)
    assert abs(o[2] - (face - 0.2)) < 1e-9 and n[2] == -1.0  # camera above: keep below

    c.set_free((1.0, 1.0, 0.0), (30.0, 20.0, 5.0))
    assert c.axis == FREE and abs(np.linalg.norm(c.free_normal) - 1.0) < 1e-12
    cells = c.cell_plane(*c.plane(torch, face, cam), np.zeros(3), 0.4)
    assert len(cells) == 4
    print("[viewer_cut] CutPlane ok")


def _ray_pick() -> None:
    from waam_twin import WAAMTwin
    from waam_twin.runtime import init_taichi
    from waam_twin.viewer.extract import ray_pick_metal

    init_taichi()
    twin = WAAMTwin(nx=24, ny=16, nz=16, dx=4e-4, max_tracers=4, enable_vof=True)
    twin.reset()
    g = twin.grid
    face_k = int(twin.nz_solid)
    out = ti.field(ti.f32, shape=4)
    # Straight down at cell (12.5, 8.5): hits the plate face.
    ray_pick_metal(g.phi, g.flags, out, 12.5, 8.5, 15.0, 0.0, 0.0, -1.0, 40.0,
                   g.FLAG_GAS, g.nx, g.ny, g.nz, 0, 0.0, 0.0, 0.0, 0.0)
    h = out.to_numpy()
    assert h[0] == 1.0 and face_k - 1 <= h[3] <= face_k, (h, face_k)
    # Horizontal ray along +x inside the plate, cut keeps x ≥ 10: first metal
    # on the kept side is the cut face at x = 10, not the plate edge at x = 0.
    z_in = face_k - 1.5
    ray_pick_metal(g.phi, g.flags, out, -3.0, 8.5, z_in, 1.0, 0.0, 0.0, 60.0,
                   g.FLAG_GAS, g.nx, g.ny, g.nz, 1, 1.0, 0.0, 0.0, 10.0)
    h = out.to_numpy()
    assert h[0] == 1.0 and 10.0 <= h[1] < 10.3, h
    ray_pick_metal(g.phi, g.flags, out, 12.5, 8.5, 15.0, 0.0, 0.0, 1.0, 40.0,
                   g.FLAG_GAS, g.nx, g.ny, g.nz, 0, 0.0, 0.0, 0.0, 0.0)
    assert out.to_numpy()[0] == 0.0  # pointing up into gas: no hit
    print(f"[viewer_cut] ray pick ok (face k={face_k}, cut face hit x={h[1]:.2f})")


def run() -> None:
    _cut_plane()
    _ray_pick()


if __name__ == "__main__":
    run()
    print("PASS")
