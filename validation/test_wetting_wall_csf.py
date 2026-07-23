"""
test_wetting_wall_csf.py — Brackbill wall-normal correction changes F direction.

PHYSICS_FORCE_CORRECTNESS_SPEC FC-6 / WP-G: ghost-φ + Young normal (no κ=2cosθ
or empirical sinθ lateral drive). Magnitude of F = γ κ |∇φ| is similar; the
corrected n̂ must rotate the force relative to wetting-off.
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
    # Slanted φ near the wall so ∇φ has a horizontal component for the
    # contact-angle correction to rotate.
    phi_np[i, j, k] = 0.7
    flags_np[i, j, k] = g.FLAG_IFACE
    if i + 1 < g.nx:
        phi_np[i + 1, j, k] = 0.35
        flags_np[i + 1, j, k] = g.FLAG_IFACE
    if k + 1 < g.nz:
        phi_np[i, j, k + 1] = 0.25
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
    Fx_off = float(g.Fx[i, j, k])
    Fy_off = float(g.Fy[i, j, k])
    Fz_off = float(g.Fz[i, j, k])
    F_off = math.sqrt(Fx_off**2 + Fy_off**2 + Fz_off**2)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
        enable_wetting=True,
        theta_rad=math.radians(30.0),  # strongly wetting → clear n̂ tilt
    )
    Fx_on = float(g.Fx[i, j, k])
    Fy_on = float(g.Fy[i, j, k])
    Fz_on = float(g.Fz[i, j, k])
    F_on = math.sqrt(Fx_on**2 + Fy_on**2 + Fz_on**2)

    dF = math.sqrt(
        (Fx_on - Fx_off) ** 2 + (Fy_on - Fy_off) ** 2 + (Fz_on - Fz_off) ** 2
    )
    print(
        f"[wetting_wall_csf] |F|_off={F_off:.3e}  |F|_on={F_on:.3e}  "
        f"|ΔF|={dF:.3e} lu"
    )
    if F_on < 1e-12 and F_off < 1e-12:
        raise AssertionError("CSF force is zero at wall-adjacent interface cell")
    if dF < 1e-14:
        raise AssertionError(
            "wetting normal correction did not change wall-adjacent force vector"
        )


if __name__ == "__main__":
    run()
    print("PASS")
