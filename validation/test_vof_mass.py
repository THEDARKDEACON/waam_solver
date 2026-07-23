"""
test_vof_mass.py — VOF metal volume conservation under uniform advection.
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def run(n_steps: int = 80, threshold_pct: float = 0.5) -> float:
    # Gate tightened 2% → 0.5%: the flux-form advection conserves φ to
    # round-off (measured drift 0.00%).
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=32, ny=32, nz=16, dx=3e-4, C_darcy=0.0, max_tracers=10, enable_vof=True)
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    phi0 = float(g.phi.to_numpy().sum())
    ux_np = g.ux.to_numpy()
    flags_np = g.flags.to_numpy()
    ux_np[flags_np == g.FLAG_FLUID] = 0.02
    g.ux.from_numpy(ux_np)

    for _ in range(n_steps):
        kernels.advect_phi(
            g.phi_tmp, g.phi, g.ux, g.uy, g.uz, g.flags,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        g.phi.copy_from(g.phi_tmp)
        kernels.reinitialize_phi(g.phi, g.flags, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_FLUID)

    phi1 = float(g.phi.to_numpy().sum())
    drift = abs(phi1 - phi0) / max(phi0, 1e-9) * 100.0
    print(f"[vof_mass] phi sum drift = {drift:.2f}%  (threshold {threshold_pct}%)")
    if drift >= threshold_pct:
        raise AssertionError(f"VOF mass drift {drift:.2f}% >= {threshold_pct}%")
    return drift


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
