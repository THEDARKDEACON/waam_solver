"""
test_marangoni_cell.py — Thermocapillary force direction vs dγ/dT sign.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.physics import forces


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=40, ny=16, nz=20, dx=2.5e-4, max_tracers=10)
    twin.reset()
    g = twin.grid

    T_np = g.T.to_numpy()
    phi_np = g.phi.to_numpy()
    fl_np = g.f_l.to_numpy()
    flags_np = g.flags.to_numpy()
    nz_s = twin.nz_solid
    j = g.ny // 2
    for i in range(g.nx):
        for k in range(g.nz):
            if k < nz_s - 1:
                phi_np[i, j, k] = 1.0
                fl_np[i, j, k] = 0.0
                flags_np[i, j, k] = g.FLAG_SOLID
            elif k == nz_s - 1:
                phi_np[i, j, k] = 0.5
                fl_np[i, j, k] = 1.0
                flags_np[i, j, k] = g.FLAG_IFACE
                T_np[i, j, k] = 300.0 + 200.0 * (i / max(g.nx - 1, 1))
            else:
                phi_np[i, j, k] = 0.0
                fl_np[i, j, k] = 0.0
                flags_np[i, j, k] = g.FLAG_GAS

    g.T.from_numpy(T_np)
    g.phi.from_numpy(phi_np)
    g.f_l.from_numpy(fl_np)
    g.flags.from_numpy(flags_np)

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_marangoni_force(
        g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
        twin.dgamma_dT_lu, g.dx,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )

    Fx = g.Fx.to_numpy()[:, j, nz_s - 1]
    dT = np.gradient(T_np[:, j, nz_s - 1])
    # ER70S-6: dγ/dT < 0 → flow from hot to cold → Fx opposes +dT/dx
    hot_side = Fx[dT > 0].mean() if np.any(dT > 0) else 0.0
    print(f"[marangoni_cell] mean Fx on hot side = {hot_side:.3e}  (expect < 0)")
    if hot_side >= 0.0:
        raise AssertionError("Marangoni force should pull from hot toward cold (Fx < 0 on +dT/dx side)")
    return float(hot_side)


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
