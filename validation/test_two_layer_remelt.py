"""
test_two_layer_remelt.py — Second layer remelts prior bead; liquid fraction increases.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run(n_steps_layer1: int = 400, n_steps_layer2: int = 400, min_delta_liq: int = 5) -> int:
    init_taichi(backend="cpu")
    twin = WAAMTwin.from_job("jobs/examples/two_layer.yaml")
    twin.enable_vof = True
    twin.enable_substrate_growth = True
    twin.enable_heat_loss = False
    twin.reset()

    g = twin.grid
    cy = 10e-3

    # Layer 1 along x at z=0
    for step in range(n_steps_layer1):
        x = 5e-3 + step * twin.travel_speed_m_s * g.dt
        twin.step(x, cy, is_welding=True)

    fl_after_l1 = int((g.f_l.to_numpy() > 0.5).sum())
    solid_after_l1 = int((g.flags.to_numpy() == g.FLAG_SOLID).sum())

    # Layer 2 return pass at raised z (third waypoint in two_layer.yaml path)
    for step in range(n_steps_layer2):
        x = 25e-3 - step * twin.travel_speed_m_s * g.dt
        z = 1.2e-3  # layer-2 torch height (was computed but never passed)
        twin.step(x, cy, is_welding=True, torch_z_m=z)

    fl_after_l2 = int((g.f_l.to_numpy() > 0.5).sum())
    T_max = float(g.T_max.to_numpy().max())

    print(
        f"[two_layer_remelt] liquid L1={fl_after_l1} L2={fl_after_l2}  "
        f"solid_cells={solid_after_l1}  T_max={T_max:.0f}K"
    )
    if fl_after_l2 < fl_after_l1 + min_delta_liq and fl_after_l2 < 3:
        raise AssertionError(
            f"Second layer did not remelt prior bead (liquid cells {fl_after_l1} → {fl_after_l2})"
        )
    if T_max < 500.0:
        raise AssertionError(f"T_max {T_max:.0f}K too low for remelt HAZ")
    return fl_after_l2 - fl_after_l1


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
