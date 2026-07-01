"""
test_wetting_wall_csf.py — Wall CSF κ = 2 cos(θ) produces substrate-adjacent force.
"""

from __future__ import annotations

import math

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.physics import forces
from waam_twin.platform import init_taichi


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=24, ny=20, nz=18, dx=3e-4,
        enable_csf_tension=True,
        enable_wetting=True,
        contact_angle_deg=70.0,
        max_tracers=10,
    )
    twin.reset()
    g = twin.grid
    nz_s = twin.nz_solid
    i, j = g.nx // 2, g.ny // 2
    k = nz_s
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    phi_np[i, j, k] = 1.0
    flags_np[i, j, k] = g.FLAG_FLUID
    if k + 1 < g.nz:
        phi_np[i, j, k + 1] = 0.4
        flags_np[i, j, k + 1] = g.FLAG_IFACE
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
        enable_wetting=False,
    )
    F_off = float(np.sqrt(g.Fx[i, j, k] ** 2 + g.Fy[i, j, k] ** 2 + g.Fz[i, j, k] ** 2))

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
        enable_wetting=True,
        theta_rad=math.radians(70.0),
    )
    F_on = float(np.sqrt(g.Fx[i, j, k] ** 2 + g.Fy[i, j, k] ** 2 + g.Fz[i, j, k] ** 2))

    print(f"[wetting_wall_csf] |F| wetting_off={F_off:.3e}  wetting_on={F_on:.3e} lu")
    if F_on < 1e-12:
        raise AssertionError("wall wetting CSF force is zero at substrate fluid cell")
    if abs(F_on - F_off) < 1e-14:
        raise AssertionError("wetting CSF should change force at wall-adjacent cell")


if __name__ == "__main__":
    run()
    print("PASS")
