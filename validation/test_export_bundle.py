"""
test_export_bundle.py — Research bundle writes volume, meta, telemetry.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from waam_twin import WAAMTwin
from waam_twin.export.bundle import write_pvd
from waam_twin.platform import init_taichi


def run() -> None:
    os.environ.pop("WAAM_HEADLESS", None)
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=18, ny=10, nz=10, dx=3e-4, max_tracers=8, enable_vof=True)
    twin.reset()
    twin.step(0.002, 0.002, is_welding=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bundle"
        paths = twin.export_research_bundle(out, tag="t0", tiers=(0, 3))
        assert "volume" in paths
        assert "telemetry" in paths
        assert "meta" in paths
        meta = json.loads(Path(paths["meta"]).read_text())
        assert meta["grid"]["nx"] == 18
        assert "unit_conversions" in meta
        print(f"[export_bundle] files={list(paths.keys())}")

        seq = Path(tmp) / "seq"
        f0 = seq / "frame_0000"
        f1 = seq / "frame_0001"
        f0.mkdir(parents=True)
        f1.mkdir(parents=True)
        v0 = f0 / "volume_step_000001.vti"
        v1 = f1 / "volume_step_000051.vti"
        v0.write_text("stub")
        v1.write_text("stub")
        pvd = seq / "sequence.pvd"
        write_pvd(pvd, [str(v0), str(v1)])
        text = pvd.read_text()
        assert 'file="frame_0000/volume_step_000001.vti"' in text
        assert "viewer_output" not in text


if __name__ == "__main__":
    run()
    print("PASS")
