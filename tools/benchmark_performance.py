"""
benchmark_performance.py — Cells/sec smoke benchmark per preset/backend.

Usage:
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.benchmark_performance
"""

from __future__ import annotations

import json
import os
import pathlib
import time

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.validation.metadata import build_run_metadata


def run(n_steps: int = 200) -> dict:
    backend = os.environ.get("WAAM_BACKEND", "cpu")
    preset = os.environ.get("WAAM_PRESET", "minimal")
    init_taichi(backend=backend)
    twin = WAAMTwin.from_preset(preset)
    twin.reset()
    g = twin.grid
    n_cells = g.nx * g.ny * g.nz
    cy = (g.ny // 2) * g.dx

    t0 = time.perf_counter()
    for step in range(n_steps):
        x = 0.01 + step * 0.005 * g.dt
        twin.step(x, cy, is_welding=True)
    elapsed = time.perf_counter() - t0
    cells_per_sec = n_cells * n_steps / max(elapsed, 1e-9)

    meta = build_run_metadata(twin, extra={
        "benchmark_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "cells_per_sec": round(cells_per_sec, 0),
    })
    return meta


def main() -> int:
    result = run()
    out = pathlib.Path("waam_twin/validation/baselines/performance_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[benchmark] {result.get('preset')} / {result.get('backend')}  "
        f"{result['cells_per_sec']:.0f} cells/s  ({result['elapsed_s']}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
