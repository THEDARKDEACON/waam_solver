"""
test_csf_toe_solid.py — CSF force nonzero at a synthetic substrate triple line.

Fixture: flat solid wall below a fluid/gas interface cell. Domain-rim skip must
not kill solid-adjacent toe cells; solid-aware normals keep κ/F defined.
"""

from __future__ import annotations

import math

from waam_twin import WAAMTwin
from waam_twin.physics import forces
from waam_twin.runtime import init_taichi


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=20, ny=16, nz=14, dx=3e-4,
        enable_csf_tension=True,
        enable_wetting=True,
        contact_angle_deg=80.0,
        max_tracers=10,
    )
    twin.reset()
    g = twin.grid
    nz_s = twin.nz_solid
    # Interior cell above substrate (not domain rim in x/y).
    i, j = g.nx // 2, g.ny // 2
    k = nz_s
    assert 1 <= i <= g.nx - 2 and 1 <= j <= g.ny - 2

    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    # Ensure substrate below is solid.
    assert flags_np[i, j, k - 1] == g.FLAG_SOLID

    # Flat-ish interface at the wall: liquid-ish centre, gas above.
    phi_np[i, j, k] = 0.55
    flags_np[i, j, k] = g.FLAG_IFACE
    if k + 1 < g.nz:
        phi_np[i, j, k + 1] = 0.15
        flags_np[i, j, k + 1] = g.FLAG_GAS
    # Mild transverse gradient so |∇φ| is nonzero after solid mirroring.
    if i + 1 < g.nx:
        phi_np[i + 1, j, k] = 0.35
        flags_np[i + 1, j, k] = g.FLAG_IFACE
    if i - 1 >= 0:
        phi_np[i - 1, j, k] = 0.70
        flags_np[i - 1, j, k] = g.FLAG_IFACE

    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
        enable_wetting=True,
        theta_rad=math.radians(80.0),
    )
    Fx = float(g.Fx[i, j, k])
    Fy = float(g.Fy[i, j, k])
    Fz = float(g.Fz[i, j, k])
    Fmag = math.sqrt(Fx * Fx + Fy * Fy + Fz * Fz)
    print(
        f"[csf_toe_solid] wall cell ({i},{j},{k})  "
        f"F=({Fx:.3e},{Fy:.3e},{Fz:.3e})  |F|={Fmag:.3e}"
    )
    if Fmag < 1e-14:
        raise AssertionError(
            "CSF force is zero at solid-adjacent triple-line cell "
            "(toe must not be skipped; solid-aware stencil required)"
        )


if __name__ == "__main__":
    run()
    print("PASS")
