"""Trilinear velocity sample matches a linear ux field at a half-index."""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=12, ny=12, nz=12, dx=1e-3, max_tracers=4)
    twin.reset()
    g = twin.grid
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    flags[:, :, :] = g.FLAG_FLUID
    fl[:, :, :] = 1.0
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)

    ux = np.zeros((g.nx, g.ny, g.nz), dtype=np.float32)
    for i in range(g.nx):
        ux[i, :, :] = float(i)
    g.ux.from_numpy(ux)
    g.uy.fill(0.0)
    g.uz.fill(0.0)

    i0, j0, k0 = 4, 5, 6
    # Cell-index 4.5 → trilinear of ux=i is 4.5 lu/ts.
    pos = np.zeros((g.max_tracers, 3), dtype=np.float32)
    active = np.zeros(g.max_tracers, dtype=np.int32)
    pos[0] = ((i0 + 0.5) * g.dx, (j0 + 0.5) * g.dx, (k0 + 0.5) * g.dx)
    active[0] = 1
    g.porosity_pos.from_numpy(pos)
    g.porosity_active.from_numpy(active)

    kernels.advect_tracers(
        g.porosity_pos, g.porosity_active,
        g.ux, g.uy, g.uz, g.f_l, g.flags,
        g.dx, g.dt, g.max_tracers,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    out = g.porosity_pos.to_numpy()[0]
    expected_x = pos[0, 0] + 4.5 * g.dx
    err = abs(float(out[0]) - expected_x)
    print(f"[tracers_trilinear] x={out[0]*1e3:.4f}mm  expected={expected_x*1e3:.4f}mm  err={err*1e6:.3f}µm")
    if err > 1e-8:
        raise AssertionError(f"trilinear sample err {err} m (want ~0 at cell center 4.5)")
    if abs(float(out[1]) - pos[0, 1]) > 1e-12 or abs(float(out[2]) - pos[0, 2]) > 1e-12:
        raise AssertionError("uy/uz should be zero")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
