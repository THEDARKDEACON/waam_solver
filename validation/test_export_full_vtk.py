"""
test_export_full_vtk.py — Full volume VTK export includes research field arrays.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi


def run(min_arrays: int = 15) -> None:
    os.environ.pop("WAAM_HEADLESS", None)
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=20, ny=12, nz=12, dx=3e-4, max_tracers=10, enable_vof=True)
    twin.reset()
    for _ in range(30):
        twin.step(0.003, 0.003, is_welding=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "full.vti")
        out = twin.export_vtk_full(path, tiers=(0, 1, 2, 3))
        assert out is not None
        try:
            import pyvista as pv
        except ImportError:
            print("[export_full_vtk] pyvista not installed — skip readback")
            return
        grid = pv.read(out)
        n = len(grid.cell_data.keys())
        print(f"[export_full_vtk] arrays={n} keys={list(grid.cell_data.keys())[:8]}...")
        assert "Velocity_Y_ms" in grid.cell_data
        assert "Curvature_kappa" in grid.cell_data
        assert n >= min_arrays


if __name__ == "__main__":
    run()
    print("PASS")
