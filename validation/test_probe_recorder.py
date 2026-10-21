"""
test_probe_recorder.py — Probe CSV records T(t) during simulation.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from waam_twin import WAAMTwin
from waam_twin.export.probes import ProbeRecorder
from waam_twin.runtime import init_taichi


def test_flush_keeps_arc_off_rows() -> None:
    """The on-disk log grows after arc_on drops. A second flush does not rewrite the weld."""
    import tempfile

    from waam_twin.export.probes import ProbeRecorder, ProbeSpec

    rec = ProbeRecorder()
    rec.probes.append(ProbeSpec("sub", 0.0, 0.0, 0.0))
    rec._rows.append({"sim_time_ms": 10.0, "step": 1, "arc_on": 1, "sub_T_K": 400.0})
    rec._rows.append({"sim_time_ms": 20.0, "step": 2, "arc_on": 1, "sub_T_K": 500.0})
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "probes.csv"
    assert rec.flush_csv(path) == 2
    rec._rows.append({"sim_time_ms": 30.0, "step": 3, "arc_on": 0, "sub_T_K": 350.0})
    assert rec.flush_csv(path) == 1
    assert rec.flush_csv(path) == 0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["arc_on"] for r in rows] == ["1", "1", "0"]
    assert rows[-1]["sim_time_ms"] == "30.0"


def test_z_ref_resolution() -> None:
    """Job z_ref resolves to domain world mm; stored probes are world-frame."""
    from waam_twin.export.probes import (
        ProbeRecorder,
        normalize_z_ref,
        resolve_probe_z_world_mm,
    )

    assert normalize_z_ref("plate-face") == "plate_face"
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=20, ny=20, nz=40, dx=1e-3, max_tracers=5)
    twin.plate_thickness_mm = 9.0
    twin.plate_center_z = True
    twin.nz_solid = twin.resolve_nz_solid()
    dx_mm = twin.grid.dx * 1000.0
    bottom = twin.plate_k0 * dx_mm
    face = twin.nz_solid * dx_mm
    assert twin.plate_k0 > 0
    assert abs(resolve_probe_z_world_mm(twin, 7.0, "plate_bottom") - (bottom + 7.0)) < 1e-9
    assert abs(resolve_probe_z_world_mm(twin, -2.0, "plate_face") - (face - 2.0)) < 1e-9
    assert abs(resolve_probe_z_world_mm(twin, 12.0, "world") - 12.0) < 1e-9

    rec = ProbeRecorder.from_job_list(
        {
            "z_ref": "plate_bottom",
            "points": [
                {"name": "mid", "x_mm": 10.0, "y_mm": 10.0, "z_mm": 7.0},
                {"name": "face_tc", "x_mm": 10.0, "y_mm": 10.0, "z_mm": -2.0, "z_ref": "plate_face"},
            ],
        },
        twin,
    )
    assert abs(rec.probes[0].z_m * 1000.0 - (bottom + 7.0)) < 1e-9
    assert abs(rec.probes[1].z_m * 1000.0 - (face - 2.0)) < 1e-9
    # Same physical depth: 2 mm under face on a 9 mm plate
    assert abs(rec.probes[0].z_m - rec.probes[1].z_m) < 1e-9
    print(
        f"[probe_z_ref] plate_k0={twin.plate_k0} face={twin.nz_solid} "
        f"mid_z={rec.probes[0].z_m*1e3:.2f} mm world"
    )



def run(min_rows: int = 5) -> None:
    test_flush_keeps_arc_off_rows()
    test_z_ref_resolution()
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=16, ny=10, nz=10, dx=3e-4, max_tracers=5)
    twin.reset()
    twin.probe_recorder = ProbeRecorder()
    twin.probe_recorder.add_grid(4, 5, 3, twin, "sub")

    for _ in range(min_rows):
        twin.step(0.002, 0.002, is_welding=True)

    twin.probe_recorder.add_grid(6, 5, 4, twin, "torch_1")
    for _ in range(3):
        twin.step(0.002, 0.002, is_welding=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probes.csv"
        twin.probe_recorder.write_csv(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= min_rows + 3
        assert "sub_T_K" in rows[0]
        assert "torch_1_T_K" in rows[0]
        assert rows[0]["torch_1_T_K"] == ""
        assert rows[-1]["torch_1_T_K"] != ""
        print(f"[probe_recorder] rows={len(rows)}  T_last={rows[-1]['sub_T_K']}")

    _run_torch_relative()


def _run_torch_relative() -> None:
    """A behind_mm probe tracks travel; a fixed probe stays put."""
    twin = WAAMTwin(nx=48, ny=48, nz=12, dx=1e-3, max_tracers=5)
    twin.reset()
    twin.probe_recorder = ProbeRecorder.from_job_list(
        [
            {"name": "fixed", "x_mm": 20.0, "y_mm": 15.0, "z_mm": 2.0},
            {"name": "trail", "behind_mm": 10.0, "lateral_mm": 0.0, "z_mm": 2.0, "x_mm": 1.0},
        ],
        twin,
    )
    trail = twin.probe_recorder.probes[1]
    assert trail.follow_torch
    assert trail.behind_m == 0.01

    # Same seeding run_path does before the first step: travel is +Y.
    twin._torch_dir_xyz = (0.0, 1.0, 0.0)
    twin.step(0.020, 0.025, is_welding=False)
    world = trail.world_m(twin)
    assert world is not None
    assert abs(world[0] - 0.020) < 1e-9
    assert abs(world[1] - 0.015) < 1e-9
    assert abs(world[2] - 0.002) < 1e-9
    assert trail.resolve(twin) == (20, 15, 2)

    twin.step(0.020, 0.030, is_welding=False)
    world = trail.world_m(twin)
    assert world is not None
    assert abs(world[0] - 0.020) < 1e-9
    assert abs(world[1] - 0.020) < 1e-9
    fixed = twin.probe_recorder.probes[0].world_m(twin)
    assert fixed == (0.020, 0.015, 0.002)

    # Left of +Y travel is −X.
    twin.probe_recorder.probes[1].lateral_m = 0.005
    world = trail.world_m(twin)
    assert world is not None
    assert abs(world[0] - 0.015) < 1e-9
    assert abs(world[1] - 0.020) < 1e-9
    print(
        f"[probe_recorder] trail=({world[0]*1e3:.1f}, {world[1]*1e3:.1f}, {world[2]*1e3:.1f}) mm"
    )


if __name__ == "__main__":
    run()
    print("PASS")
