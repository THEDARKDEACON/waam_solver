"""C1 regression: melted substrate cells must be able to flow (nonzero |u|)."""

from __future__ import annotations

import os

import numpy as np


def run() -> float:
    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(nx=32, ny=24, nz=20, dx=4e-4, max_tracers=10, C_darcy=0.0)
    twin.enable_vof = True
    twin.enable_marangoni = True
    twin.reset()
    g = twin.grid
    nz = twin.nz_solid

    # Melt a block of substrate under a lateral T gradient
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    fl = g.f_l.to_numpy()
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    T_liq = twin.mat.T_liquidus
    for i in range(8, 24):
        for j in range(4, 20):
            for k in range(max(1, nz - 4), nz):
                T[i, j, k] = T_liq + 50.0 + 30.0 * (j / g.ny)
                H[i, j, k] = twin.H_liq + twin.cp_rho * 50.0
                fl[i, j, k] = 1.0
                phi[i, j, k] = 1.0
                flags[i, j, k] = g.FLAG_FLUID
    g.T.from_numpy(T)
    g.H.from_numpy(H)
    g.f_l.from_numpy(fl)
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)

    for _ in range(40):
        twin.step(0.5 * g.nx * g.dx, 0.5 * g.ny * g.dx, is_welding=False)

    fl = g.f_l.to_numpy()
    flags = g.flags.to_numpy()
    u = np.sqrt(
        g.ux.to_numpy() ** 2 + g.uy.to_numpy() ** 2 + g.uz.to_numpy() ** 2
    )
    kk = np.arange(g.nz)[None, None, :]
    sub_liq = (fl > 0.5) & (kk < nz) & (flags == g.FLAG_FLUID)
    n = int(sub_liq.sum())
    umax = float(u[sub_liq].max()) if n else 0.0
    print(f"[substrate_flow] n_sub_liq={n}  max|u|_lu={umax:.3e}")
    if n < 1:
        raise AssertionError("C1: no liquid FLUID cells inside substrate band")
    if umax < 1e-8:
        raise AssertionError(
            "C1: substrate liquid velocity is zero — flags still force SOLID"
        )
    return umax


if __name__ == "__main__":
    run()
