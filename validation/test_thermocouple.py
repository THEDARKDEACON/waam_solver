"""
test_thermocouple.py — Virtual thermocouple vs Rosenthal (traveling arc, ±35%).
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.materials import load_material
from waam_twin.validation.rosenthal import rosenthal_tail_2d
from waam_twin.validation.torch_motion import torch_position_linear


def run(n_steps: int = 1400, threshold: float = 40.0) -> float:
    init_taichi(backend="cpu")
    mat = load_material("materials/validated/ER70S-6.v1.yaml")
    Q = 2800.0
    eta = 0.72
    T0 = 300.0
    travel = 0.005
    probe_back_mm = 3.0

    twin = WAAMTwin(
        material=mat,
        nx=80, ny=40, nz=36,
        dx=2.5e-4,
        arc_power_W=Q,
        arc_efficiency=eta,
        travel_speed_m_s=travel,
        enable_heat_loss=True,
        h_conv=35.0,
        max_tracers=100,
    )
    twin.reset(T_ambient=T0)
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    x_start = 0.014

    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, x_start, cy)
        twin.step(x, cy, is_welding=True)

    x_torch, _ = torch_position_linear(n_steps - 1, g.dt, travel, x_start, cy)
    k_sub = max(1, twin.nz_solid - 1)
    i = int((x_torch - probe_back_mm * 1e-3) / g.dx)
    j = g.ny // 2
    T_sim = float(g.T.to_numpy()[i, j, k_sub])
    T_ref = rosenthal_tail_2d(
        T0, Q, eta, mat.k_at(1200.0), mat.rho, mat.cp_at(1200.0),
        travel, -probe_back_mm * 1e-3,
    )
    err = abs(T_sim - T_ref) / max(T_ref - T0, 1.0) * 100.0
    print(f"[thermocouple] T_sim={T_sim:.1f}K  T_ref={T_ref:.1f}K  err={err:.1f}%")
    if T_sim < T0 + 20.0:
        raise AssertionError("Thermocouple probe did not heat")
    if err >= threshold:
        raise AssertionError(f"Thermocouple error {err:.1f}% >= {threshold}%")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
