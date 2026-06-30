"""
test_soak_10k.py — Long-run stability: 10k steps without NaN, mass conserved.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.validation.torch_motion import advance_torch


def run(n_steps: int = 10_000, mass_threshold_pct: float = 2.0) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=48, ny=24, nz=24,
        dx=3e-4,
        arc_power_W=2500.0,
        travel_speed_m_s=0.004,
        enable_heat_loss=True,
        h_conv=25.0,
        max_tracers=200,
    )
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    x = 0.008
    x_min = 0.005
    x_max = (g.nx - 5) * g.dx
    rho0 = float(g.rho.to_numpy().mean())

    for step in range(n_steps):
        twin.step(x, cy, is_welding=True)
        x = advance_torch(x, twin.travel_speed_m_s, g.dt, x_min, x_max)
        if step % 2000 == 1999:
            T = g.T.to_numpy()
            if not np.isfinite(T).all():
                raise AssertionError(f"NaN/Inf in T at step {step + 1}")

    rho_mean = float(g.rho.to_numpy().mean())
    drift = abs(rho_mean - rho0) / rho0 * 100.0
    print(f"[soak_10k] {n_steps} steps OK  rho drift={drift:.3f}%  (threshold {mass_threshold_pct}%)")
    if drift >= mass_threshold_pct:
        raise AssertionError(f"Mass drift {drift:.3f}% >= {mass_threshold_pct}%")
    return drift


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
