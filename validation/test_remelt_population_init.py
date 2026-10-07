"""
test_remelt_population_init.py — SOLID→FLUID remelt must reset LBM populations.

Poison a solid cell's f with large values, raise H above liquidus, run remelt,
and assert ρ≈1 and |u|≈0 (quiescent equilibrium), not the poisoned moments.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.physics import free_surface
from waam_twin.runtime import init_taichi


def run() -> float:
    init_taichi(backend="cpu")
    n = 16
    twin = WAAMTwin(
        nx=n, ny=n, nz=n, dx=5e-4,
        enable_vof=True, enable_bead_freeze=True,
        C_darcy=0.0, max_tracers=10,
    )
    twin.reset(test_fluid_domain=True)
    g = twin.grid
    # Make one interior cell solid with poisoned populations
    i = j = k = n // 2
    flags = g.flags.to_numpy()
    flags[i, j, k] = g.FLAG_SOLID
    g.flags.from_numpy(flags)
    fl = g.f_l.to_numpy()
    fl[i, j, k] = 0.0
    g.f_l.from_numpy(fl)
    phi = g.phi.to_numpy()
    phi[i, j, k] = 1.0
    g.phi.from_numpy(phi)
    # Poison f_src (post-stream buffer name)
    f = g.f_src.to_numpy()
    f[:, i, j, k] = 10.0  # nonsense
    g.f_src.from_numpy(f)
    H = g.H.to_numpy()
    H[i, j, k] = float(twin.H_liq + 0.1 * twin.L_rho)
    g.H.from_numpy(H)

    free_surface.remelt_hot_solid(
        g.T, g.H, g.f_l, g.phi, g.flags,
        g.f_src, g.rho, g.ux, g.uy, g.uz, g.w,
        twin.L_rho, twin.H_sol, twin.H_liq,
        g.FLAG_SOLID, g.FLAG_FLUID,
    )

    flags2 = g.flags.to_numpy()
    if int(flags2[i, j, k]) != g.FLAG_FLUID:
        raise AssertionError("remelt did not open SOLID→FLUID")
    rho = float(g.rho.to_numpy()[i, j, k])
    ux = float(g.ux.to_numpy()[i, j, k])
    uy = float(g.uy.to_numpy()[i, j, k])
    uz = float(g.uz.to_numpy()[i, j, k])
    f2 = g.f_src.to_numpy()[:, i, j, k]
    mass = float(f2.sum())
    if abs(rho - 1.0) > 1e-5 or abs(mass - 1.0) > 1e-4:
        raise AssertionError(f"bad remelt ρ/mass: rho={rho} mass={mass}")
    umag = (ux * ux + uy * uy + uz * uz) ** 0.5
    if umag > 1e-6:
        raise AssertionError(f"remelt left nonzero velocity |u|={umag}")
    if float(np.max(np.abs(f2))) > 1.0:
        raise AssertionError("remelt left poisoned population magnitudes")
    print(f"[remelt_population_init] rho={rho:.4f} |u|={umag:.2e} mass={mass:.4f} PASS")
    return umag


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    raise SystemExit(0)
