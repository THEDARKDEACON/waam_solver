"""
test_multi_bead.py — Torch path driver smoke test (multi_bead job).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run(n_steps: int = 600) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin.from_job("jobs/examples/multi_bead.yaml")
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.reset()
    twin.run_path("jobs/examples/multi_bead.yaml", n_steps=n_steps, is_welding=True)

    T_max = float(twin.grid.T_max.to_numpy().max())
    T_amb = twin.T_amb
    print(f"[multi_bead] steps={n_steps}  T_max={T_max:.0f}K  (ambient {T_amb:.0f}K)")
    if T_max < T_amb + 50.0:
        raise AssertionError("Path weld did not heat substrate")
    return T_max - T_amb


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
