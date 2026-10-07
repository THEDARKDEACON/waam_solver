"""Mach clamp must resync post-collision populations with clamped u."""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi
from waam_twin import kernels
from waam_twin.lattice import EX, EY, EZ, W


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=12, ny=12, nz=12, dx=5e-4,
        C_darcy=0.0, max_tracers=10,
        force_limit_lu=10.0, u_mach_limit_lu=0.05,
        use_srt=True, use_variable_tau=False,
    )
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    # Drive one cell well above the Mach cap, rebuild feq, then clamp.
    i = j = k = 6
    ux0, uy0, uz0 = 0.20, 0.0, 0.0
    g.ux[i, j, k] = ux0
    g.uy[i, j, k] = uy0
    g.uz[i, j, k] = uz0
    g.rho[i, j, k] = 1.0
    r = 1.0
    u2 = ux0 * ux0
    for q in range(19):
        eu = EX[q] * ux0
        g.f_dst[q, i, j, k] = W[q] * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)

    kernels.clamp_velocity_mach(
        g.f_dst, g.rho, g.ux, g.uy, g.uz, g.flags, 0.05,
        g.clamp_mach_hits_buf, g.FLAG_SOLID, g.FLAG_GAS,
    )
    n_hit = int(g.clamp_mach_hits_buf[None])
    if n_hit < 1:
        raise AssertionError("expected at least one Mach-clamp hit")

    ux = float(g.ux[i, j, k])
    uy = float(g.uy[i, j, k])
    uz = float(g.uz[i, j, k])
    mag = (ux * ux + uy * uy + uz * uz) ** 0.5
    if mag > 0.05 + 1e-6:
        raise AssertionError(f"|u|={mag} still above cap")

    f = np.array([float(g.f_dst[q, i, j, k]) for q in range(19)])
    rho = float(f.sum())
    jx = float(sum(f[q] * EX[q] for q in range(19)))
    jy = float(sum(f[q] * EY[q] for q in range(19)))
    jz = float(sum(f[q] * EZ[q] for q in range(19)))
    err = abs(jx - rho * ux) + abs(jy - rho * uy) + abs(jz - rho * uz)
    rel = err / max(abs(rho), 1e-12)
    print(f"[mach_clamp_sync] hits={n_hit} |u|={mag:.4f} mom_rel_err={rel:.2e}")
    if rel > 1e-5:
        raise AssertionError(f"population momentum desynced from u: rel={rel}")


if __name__ == "__main__":
    run()
    print("PASS")
