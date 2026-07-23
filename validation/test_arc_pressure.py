"""
test_arc_pressure.py — Arc pressure downward force + Lin–Eagar I² scaling.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.physics import forces
from waam_twin.physics.weld_forces import lin_eagar_peak_pa, arc_pressure_peak_pa


def run(min_delta_Fz: float = 1e-10) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=40, ny=20, nz=24, dx=2.5e-4,
        arc_pressure_pa=50_000.0,
        welding_current_A=200.0,
        max_tracers=10,
    )
    twin.arc_pressure_model = "constant"
    twin.reset()
    g = twin.grid
    nz_s = twin.nz_solid
    j = g.ny // 2
    k = nz_s - 1

    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    phi_np[:, j, k] = 0.5
    flags_np[:, j, k] = g.FLAG_IFACE
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    Fz_before = g.Fz.to_numpy().copy()
    forces.apply_arc_pressure(
        g.Fz, g.flags, g.phi,
        g.nx // 2, j, k, twin.sigma_cells,
        twin.arc_pressure, g.dt, g.dx, twin.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    delta = float((g.Fz.to_numpy() - Fz_before)[:, j, k].min())
    print(f"[arc_pressure] min ΔFz = {delta:.3e}  (expect < 0)")
    if delta > -min_delta_Fz:
        raise AssertionError("Arc pressure did not deflect free surface downward")

    # Lin–Eagar: doubling I → 4× peak pressure
    twin.arc_pressure_model = "lin_eagar"
    twin.welding_current_A = 100.0
    p100 = arc_pressure_peak_pa(twin)
    twin.welding_current_A = 200.0
    p200 = arc_pressure_peak_pa(twin)
    ratio = p200 / max(p100, 1e-30)
    p_analytic = lin_eagar_peak_pa(200.0, twin.arc_sigma_m)
    print(
        f"[arc_pressure] Lin–Eagar p(100A)={p100:.1f} Pa  p(200A)={p200:.1f} Pa  "
        f"ratio={ratio:.3f}  (expect 4)  analytic_200={p_analytic:.1f}"
    )
    if abs(ratio - 4.0) > 1e-9:
        raise AssertionError(f"Lin–Eagar peak must scale as I² (ratio={ratio})")
    if abs(p200 - p_analytic) > 1e-6 * max(p_analytic, 1.0):
        raise AssertionError("arc_pressure_peak_pa disagrees with lin_eagar_peak_pa")
    return delta


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
