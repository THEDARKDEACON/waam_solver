"""
test_kuka_trajectory_bridge.py — Frame mapping, CSV export, torch Z → arc_k.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from waam_twin.kuka_adapter import tcp_mm_to_sim_m
from waam_twin.frame import load_weld_frame
from waam_twin.toolpath.export import segments_to_csv_file, segments_to_waypoints
from waam_twin.platform import init_taichi


def test_frame_tcp_mapping() -> None:
    frame = load_weld_frame("jobs/frames/weld_table.yaml")
    x, y, z = tcp_mm_to_sim_m([100.0, 200.0, 5.0], frame)
    assert abs(x - 0.1) < 1e-9
    assert abs(y - 0.2) < 1e-9
    assert abs(z - 0.005) < 1e-9


def test_segments_to_csv() -> None:
    segs = [
        {"torch_state": "TRUE", "points": [[5, 10, 0], [20, 10, 0], [35, 10, 0]]},
        {"torch_state": "FALSE", "points": [[35, 10, 0], [40, 10, 0]]},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "line.csv"
        segments_to_csv_file(segs, path, weld_only=True)
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 3
        assert float(rows[0]["x_mm"]) == 5.0
    wps = segments_to_waypoints(segs, weld_only=True)
    assert len(wps) == 3
    assert abs(wps[1][0] - 0.02) < 1e-9


def test_torch_z_raises_arc_k() -> None:
    from waam_twin import WAAMTwin
    from waam_twin.solvers.coupled_step import _resolve_arc_k

    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=24, ny=16, nz=22, dx=3e-4, max_tracers=10)
    twin.use_torch_z = True
    twin.substrate_z_m = 0.0
    twin.ctwd_m = 0.015
    twin.nz_solid = 4
    twin.reset()
    g = twin.grid
    k_flat = _resolve_arc_k(twin, g, 12.0, 8.0, None)
    k_z = _resolve_arc_k(twin, g, 12.0, 8.0, 0.020)
    assert k_z >= k_flat, f"torch Z should not lower arc_k: {k_z} vs {k_flat}"


def run() -> None:
    test_frame_tcp_mapping()
    print("[kuka_bridge] frame TCP mapping OK")
    test_segments_to_csv()
    print("[kuka_bridge] segments → CSV OK")
    test_torch_z_raises_arc_k()
    print("[kuka_bridge] torch Z arc_k OK")


if __name__ == "__main__":
    run()
    print("PASS")
