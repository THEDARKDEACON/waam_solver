"""
test_surface_vtk.py — φ isosurface export produces a non-empty mesh.
"""

from __future__ import annotations

import os
import sys
import tempfile

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run() -> int:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=48, ny=24, nz=24, dx=3e-4, max_tracers=20, enable_vof=True)
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(400):
        twin.step(0.01 + step * 0.004 * g.dt, cy, is_welding=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "surf.vtp")
        twin.export_surface_vtk(path)
        try:
            import pyvista as pv
            mesh = pv.read(path)
            n = mesh.n_cells
        except Exception as exc:
            raise AssertionError(f"Could not read surface VTK: {exc}") from exc

    print(f"[surface_vtk] cells={n}")
    if n < 1:
        raise AssertionError("Empty surface mesh")
    return n


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
