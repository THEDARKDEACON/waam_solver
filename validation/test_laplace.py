"""
test_laplace.py — Capillary pressure smoke (CSF produces interface force).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.physics import forces


def run(min_force_lu: float = 1e-10) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=32, ny=32, nz=20, dx=3e-4, enable_csf_tension=True, max_tracers=10)
    twin.reset()
    g = twin.grid

    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    nz_s = twin.nz_solid
    for i in range(g.nx):
        for j in range(g.ny):
            for k in range(g.nz):
                if k < nz_s - 1:
                    phi_np[i, j, k] = 1.0
                    flags_np[i, j, k] = g.FLAG_SOLID if k < nz_s - 2 else g.FLAG_FLUID
                elif k == nz_s - 1:
                    phi_np[i, j, k] = 0.5
                    flags_np[i, j, k] = g.FLAG_IFACE
                else:
                    phi_np[i, j, k] = 0.0
                    flags_np[i, j, k] = g.FLAG_GAS
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )

    Fmag = np.sqrt(
        g.Fx.to_numpy() ** 2 + g.Fy.to_numpy() ** 2 + g.Fz.to_numpy() ** 2
    ).max()
    print(f"[laplace] max |F_csf| = {Fmag:.3e} lu  (min {min_force_lu:.1e})")
    if Fmag < min_force_lu:
        raise AssertionError("CSF tension force negligible — check phi gradient")
    return float(Fmag)


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
