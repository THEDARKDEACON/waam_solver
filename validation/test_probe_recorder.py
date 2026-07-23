"""
test_probe_recorder.py — Probe CSV records T(t) during simulation.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from waam_twin import WAAMTwin
from waam_twin.export.probes import ProbeRecorder
from waam_twin.platform import init_taichi


def run(min_rows: int = 5) -> None:
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


if __name__ == "__main__":
    run()
    print("PASS")
