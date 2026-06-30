"""
test_arc_pressure.py — Arc pressure produces downward interface force.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.physics import forces


def run(min_delta_Fz: float = 1e-10) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=40, ny=20, nz=24, dx=2.5e-4,
        arc_pressure_pa=50_000.0,
        max_tracers=10,
    )
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
    return delta


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
