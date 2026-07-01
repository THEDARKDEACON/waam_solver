"""
test_wetting_droplet.py — Contact-angle φ BC promotes wetting film on substrate.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import free_surface


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=32, ny=32, nz=24, dx=3e-4,
        enable_vof=True, enable_csf_tension=True, enable_wetting=True,
        contact_angle_deg=75.0, max_tracers=10,
    )
    twin.reset()
    g = twin.grid
    nz_s = twin.nz_solid
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    # Liquid bump on substrate centre
    ci, cj = g.nx // 2, g.ny // 2
    for di in range(-3, 4):
        for dj in range(-3, 4):
            i, j = ci + di, cj + dj
            if 0 <= i < g.nx and 0 <= j < g.ny:
                k = nz_s
                if i < g.nx and j < g.ny:
                    phi_np[i, j, k] = 1.0
                    flags_np[i, j, k] = g.FLAG_FLUID
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)

    free_surface.apply_contact_angle_phi_bc(
        g.phi, g.flags, twin.theta_rad,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    phi = g.phi.to_numpy()
    film = 0.5 * (1.0 - np.cos(twin.theta_rad))
    # Gas cells adjacent to pool should pick up wetting precursor φ
    spread = phi[ci + 4, cj, nz_s]
    assert spread >= film * 0.5, f"wetting film too weak: φ={spread:.3f} expected >={film*0.5:.3f}"
    print(f"[wetting_droplet] precursor φ={spread:.3f}  film_target={film:.3f}")


if __name__ == "__main__":
    run()
    print("PASS")
