"""
test_enthalpy_cap.py — Peak temperature stays below vaporization cap.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi


def run(max_peak_K: float = 3250.0, n_steps: int = 400) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=24, ny=16, nz=16, dx=3e-4,
        enable_vof=True,
        enable_enthalpy_cap=True,
        T_vapor_cap_K=3200.0,
        arc_surface_weighting=True,
        heat_source="goldak",
        max_tracers=10,
    )
    twin.reset()
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x = 0.003 + step * g.dt * 0.008
        twin.step(x, cy, is_welding=True)
    peak = float(g.T.to_numpy().max())
    print(f"[enthalpy_cap] T_peak={peak:.0f}K  cap={twin.T_vapor_cap_K:.0f}K")
    if peak > max_peak_K:
        raise AssertionError(f"Peak T {peak:.0f}K exceeds cap band {max_peak_K:.0f}K")
    return peak


if __name__ == "__main__":
    run()
    print("PASS")
