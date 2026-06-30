"""
test_interpass_haz.py — Two-segment path with interpass; T_max accumulates along weld.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run(n_steps: int = 500, min_Tmax_K: float = 600.0) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin.from_job("jobs/examples/two_layer.yaml")
    twin.enable_vof = False
    twin.enable_heat_loss = True
    twin.reset()
    twin.run_path("jobs/examples/two_layer.yaml", n_steps=n_steps, interpass_steps=80)

    T_max = twin.grid.T_max.to_numpy()
    flags = twin.grid.flags.to_numpy()
    metal = flags != twin.grid.FLAG_GAS
    peak = float(T_max[metal].max()) if metal.any() else float(T_max.max())

    print(f"[interpass_haz] T_max peak = {peak:.0f}K  (min {min_Tmax_K}K)")
    if peak < min_Tmax_K:
        raise AssertionError(f"HAZ T_max {peak:.0f}K below {min_Tmax_K}K")
    return peak


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
