"""
test_deposition_no_column.py — Surface deposition must not fill tall gas columns.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import deposition


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=28, ny=20, nz=22, dx=3e-4,
        enable_vof=True, droplet_freq_hz=44.0, max_tracers=10,
    )
    twin.wire_feed_m_s = 8.0 / 60.0
    twin.reset()
    g = twin.grid
    i_arc = g.nx // 3
    j_arc = g.ny // 2
    k_arc = float(twin.nz_solid)
    # Small pool on substrate
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ii, jj = i_arc + di, j_arc + dj
            kk = twin.nz_solid
            if 0 <= ii < g.nx and 0 <= jj < g.ny:
                g.phi[ii, jj, kk] = 1.0
                g.f_l[ii, jj, kk] = 1.0
                g.flags[ii, jj, kk] = g.FLAG_FLUID
                g.T[ii, jj, kk] = twin.mat.T_liquidus + 200.0

    drop_r = deposition.droplet_radius_cells(twin)
    drop_vol = deposition.droplet_mass_kg(twin) / twin.mat.rho
    foot = deposition.deposition_footprint_cells(twin)
    g.deposit_vol_buf[None] = 0.0
    g.deposit_real_buf[None] = 0.0
    deposition.feed_wire_surface(
        g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
        float(i_arc), float(j_arc), k_arc,
        foot, drop_r, drop_vol,
        twin.mat.T_liquidus + 500.0,
        twin.cp_rho, twin.L_rho, 1.0,
        g.deposit_vol_buf, g.deposit_real_buf, g.dx ** 3,
        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
        g.nx, g.ny, g.nz,
    )

    flags = g.flags.to_numpy()
    fluid_z = np.where(flags == g.FLAG_FLUID)[2]
    if fluid_z.size == 0:
        raise AssertionError("no fluid deposited")
    max_above = int(fluid_z.max() - twin.nz_solid)
    limit = int(drop_r + 4)
    assert max_above <= limit, (
        f"deposition column too tall: {max_above} cells above substrate (limit {limit})"
    )
    print(f"[deposition_no_column] max_above_substrate={max_above} cells (limit {limit})")


if __name__ == "__main__":
    run()
    print("PASS")
