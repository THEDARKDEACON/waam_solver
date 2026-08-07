"""
test_rosenthal_farfield.py — Far-field tail vs 3D Rosenthal (traveling arc).

Uses a compact Gaussian (σ ≪ probe lag) and enough physical travel for a
quasi-steady trail. Absolute match to a point-source analytic is only fair
outside the source footprint.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin.materials import load_material
from waam_twin.validation.rosenthal import rosenthal_tail_2d
from waam_twin.validation.torch_motion import torch_position_linear


def run(n_steps: int = 8000, threshold: float = 55.0) -> float:
    init_taichi(backend="cpu")
    mat = load_material("materials/placeholders/ER70S-6.yaml")
    # Faster travel → more trail length in CI time; centerline Rosenthal is
    # independent of v (ηQ / 2πk|x|).
    travel = 0.04
    Q = 2800.0
    eta = 0.72
    T0 = 300.0

    twin = WAAMTwin(
        material=mat,
        nx=96, ny=40, nz=32,
        dx=2.5e-4,
        arc_power_W=Q,
        arc_efficiency=eta,
        travel_speed_m_s=travel,
        enable_heat_loss=False,
        arc_surface_weighting=True,
        arc_sigma_mm=0.8,
        arc_penetration_mm=0.5,
        C_darcy=1.6e5,
        max_tracers=100,
    )
    twin.reset(T_ambient=T0)
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    x_start = 0.006

    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, x_start, cy)
        twin.step(x, cy, is_welding=True)

    x_torch, _ = torch_position_linear(n_steps - 1, g.dt, travel, x_start, cy)
    T_np = g.T.to_numpy()
    flags_np = g.flags.to_numpy()
    k_sub = max(1, twin.nz_solid - 1)
    errs = []
    # Probe behind the arc footprint (σ=0.8 mm → use ≥3 mm lag).
    for mm_back in (3.0, 4.0, 5.0, 6.0):
        i = int((x_torch - mm_back * 1e-3) / g.dx)
        j = g.ny // 2
        if i < 2 or flags_np[i, j, k_sub] == g.FLAG_GAS:
            continue
        T_sim = float(T_np[i, j, k_sub])
        T_anal = rosenthal_tail_2d(T0, Q, eta, mat.k, mat.rho, mat.cp, travel, -mm_back * 1e-3)
        if T_anal > T0 + 5.0 and T_sim > T0 + 1.0:
            errs.append(abs(T_sim - T_anal) / (T_anal - T0) * 100.0)

    if not errs:
        raise AssertionError("No Rosenthal tail samples")

    mean_err = float(np.mean(errs))
    print(
        f"[rosenthal_farfield] mean tail error = {mean_err:.1f}%  "
        f"(threshold {threshold}%)  travel={n_steps * travel * g.dt * 1e3:.1f}mm"
    )
    if mean_err >= threshold:
        raise AssertionError(f"Rosenthal tail error {mean_err:.1f}% >= {threshold}%")
    return mean_err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
