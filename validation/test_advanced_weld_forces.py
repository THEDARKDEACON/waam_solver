"""
test_advanced_weld_forces.py — Recoil CC, gas shear, Lorentz, droplet impact smoke tests.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import weld_forces


def run() -> None:
    init_taichi(backend="cpu")
    base_kw = dict(
        nx=24, ny=14, nz=16, dx=3e-4,
        enable_vof=True, max_tracers=10,
        wire_diameter_mm=1.2,
        travel_speed_m_s=0.008,
        droplet_freq_hz=44.0,
    )

    # Recoil CC increases downward Fz on hot surface vs cold
    twin = WAAMTwin(
        **base_kw,
        enable_recoil=True,
        use_recoil_clausius_clapeyron=True,
        enable_csf_tension=True,
        T_boiling_K=1800.0,
    )
    twin.wire_feed_m_s = 8.0 / 60.0
    twin.reset()
    g = twin.grid
    j, k = g.ny // 2, g.nz // 2
    i_arc = g.nx // 3
    k_arc = twin.nz_solid
    g.T[i_arc, j, k_arc] = 2500.0
    g.phi[i_arc, j, k_arc] = 0.8
    g.f_l[i_arc, j, k_arc] = 0.9
    g.flags[i_arc, j, k_arc] = g.FLAG_FLUID
    Fz0 = float(g.Fz[i_arc, j, k_arc])
    weld_forces.apply_recoil(twin, g, float(i_arc), float(j), float(k_arc))
    assert float(g.Fz[i_arc, j, k_arc]) < Fz0, "CC recoil should add downward force"
    print(f"[recoil_cc] ΔFz={float(g.Fz[i_arc,j,k_arc])-Fz0:.3e}")

    # Gas shear adds outward Fx/Fy
    twin2 = WAAMTwin(**base_kw, enable_gas_shear=True, gas_jet_velocity_m_s=15.0)
    twin2.wire_feed_m_s = 8.0 / 60.0
    twin2.reset()
    g2 = twin2.grid
    from waam_twin.physics import forces
    forces.clear_forces(g2.Fx, g2.Fy, g2.Fz)
    g2.phi[i_arc, j, k_arc] = 0.55
    g2.phi[i_arc + 1, j, k_arc] = 0.45
    g2.f_l[i_arc, j, k_arc] = 1.0
    g2.flags[i_arc, j, k_arc] = g2.FLAG_FLUID
    g2.f_l[i_arc + 1, j, k_arc] = 1.0
    g2.flags[i_arc + 1, j, k_arc] = g2.FLAG_FLUID
    Fx0 = float(g2.Fx[i_arc + 1, j, k_arc])
    weld_forces.apply_gas_shear(twin2, g2, float(i_arc), float(j), float(k_arc))
    assert abs(float(g2.Fx[i_arc + 1, j, k_arc]) - Fx0) > 1e-12
    print(f"[gas_shear] ΔFx={float(g2.Fx[i_arc+1,j,k_arc])-Fx0:.3e}")

    # Lorentz produces non-zero J×B body force in a molten patch
    twin3 = WAAMTwin(
        **base_kw,
        enable_lorentz=True,
        welding_current_A=180.0,
        lorentz_jacobi_iters=40,
    )
    twin3.wire_feed_m_s = 8.0 / 60.0
    twin3.reset()
    g3 = twin3.grid
    from waam_twin.physics import forces
    forces.clear_forces(g3.Fx, g3.Fy, g3.Fz)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(0, 3):
                ii, jj, kk = i_arc + di, j + dj, k_arc + dk
                if 0 <= ii < g3.nx and 0 <= jj < g3.ny and 0 <= kk < g3.nz:
                    g3.f_l[ii, jj, kk] = 0.85
                    g3.T[ii, jj, kk] = 2200.0
                    g3.flags[ii, jj, kk] = g3.FLAG_FLUID
    weld_forces.solve_lorentz(twin3, g3, float(i_arc), float(j), float(k_arc))
    fmag = np.sqrt(g3.Fx.to_numpy() ** 2 + g3.Fy.to_numpy() ** 2 + g3.Fz.to_numpy() ** 2)
    jmag = np.sqrt(g3.Jx.to_numpy() ** 2 + g3.Jy.to_numpy() ** 2 + g3.Jz.to_numpy() ** 2)
    assert float(jmag.max()) > 1.0
    assert float(fmag.max()) > 1e-12
    print(f"[lorentz] |J|_max={float(jmag.max()):.3e} |F|_max={float(fmag.max()):.3e}")

    # Droplet impact velocity scales with wire feed
    t_low = WAAMTwin(**base_kw)
    t_low.wire_feed_m_s = 4.0 / 60.0
    t_high = WAAMTwin(**base_kw)
    t_high.wire_feed_m_s = 12.0 / 60.0
    v_low = weld_forces.droplet_impact_velocity_m_s(t_low)
    v_high = weld_forces.droplet_impact_velocity_m_s(t_high)
    assert v_high >= v_low
    print(f"[droplet_impact] v_low={v_low:.3f} v_high={v_high:.3f} m/s")


if __name__ == "__main__":
    run()
    print("PASS")
