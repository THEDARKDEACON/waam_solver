"""
test_arc_surface_weight.py — Shallow solid heats more than deep solid (penetration δ).
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi


def run(n_steps: int = 120) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=24, ny=12, nz=20, dx=3e-4,
        enable_vof=False,
        arc_surface_weighting=True,
        arc_penetration_mm=2.0,
        enable_enthalpy_cap=True,
        heat_source="gaussian2d",
        max_tracers=5,
    )
    twin.reset()
    g = twin.grid
    j = g.ny // 2
    i_arc = g.nx // 3
    k_arc = twin.nz_solid - 1
    k_deep = max(0, k_arc - 6)
    H_before = g.H.to_numpy().copy()
    for step in range(n_steps):
        x = (i_arc * g.dx) + step * g.dt * 0.002
        twin.step(x, j * g.dx, is_welding=True)
    dH = g.H.to_numpy() - H_before
    shallow_gain = float(dH[i_arc, j, k_arc])
    deep_gain = float(dH[i_arc, j, k_deep])
    print(
        f"[arc_surface_weight] ΔH shallow(k={k_arc})={shallow_gain:.2e}  "
        f"deep(k={k_deep})={deep_gain:.2e}"
    )
    if shallow_gain <= deep_gain:
        raise AssertionError("Shallow solid should gain more enthalpy than deep solid")
    return shallow_gain / max(deep_gain, 1e-12)


if __name__ == "__main__":
    run()
    print("PASS")
