"""C4 regression: wetting BC must not invent metal volume."""

from __future__ import annotations

import os

import numpy as np


def run() -> float:
    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.physics import free_surface

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(nx=40, ny=40, nz=24, dx=4e-4, max_tracers=10)
    twin.enable_vof = True
    twin.enable_wetting = True
    twin.reset()
    g = twin.grid

    def phi_mass() -> float:
        phi = g.phi.to_numpy()
        flags = g.flags.to_numpy()
        return float((phi * (flags != g.FLAG_GAS)).sum())

    m0 = phi_mass()
    for _ in range(200):
        free_surface.apply_contact_angle_phi_bc(
            g.phi, g.flags, twin.theta_rad,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        free_surface.update_flags_from_phi(
            g.phi, g.f_l, g.flags, twin.nz_solid,
            g.FLAG_FLUID, g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_IFACE,
        )
    m1 = phi_mass()
    rel = abs(m1 - m0) / max(m0, 1.0)
    print(f"[wetting_no_volume] m0={m0:.1f} m1={m1:.1f} rel={rel:.3e}")
    if rel > 1e-6:
        raise AssertionError(
            f"C4: wetting BC changed metal volume by {rel:.2%} (must be zero)"
        )
    return rel


if __name__ == "__main__":
    run()
