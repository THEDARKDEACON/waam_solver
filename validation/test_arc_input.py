"""
test_arc_input.py — arc heat input for the Park physics check.

1. heat_source: surface puts exactly η·Q·dt into the workpiece, only into the
   top metal of each column (column share φ_k − φ_{k+1}), with a horizontal
   Gaussian falloff; on a partly filled interface too (fill-weighted energy).
2. energy_split: the arc gives up the running mean of the droplet power, so
   over many droplets arc + droplets = η·Q (the old per-step subtraction
   removed ~1 %), and the arc power never goes negative.
3. Pulsed current: I_mean / I_rms of Park's 345/35 A, 1.2 ms, 151.5 Hz train
   (91.4 / 150.5 A); I_rms drives the I² loads; mean power = V·I_mean +
   R·(I_rms² − I_mean²) for an arc voltage U = U0 + R·I.
4. surface_height_at returns the top metal cell of a tall column (the k loop
   is parallel; a plain assignment raced and gave 0/32/64 on the GPU, which
   put the droplet search box inside the plate at dx 0.2).
"""

from __future__ import annotations

import math

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi


def _energy(twin) -> float:
    g = twin.grid
    H = g.H.to_numpy().astype(np.float64)
    fl = g.flags.to_numpy()
    fill = np.where(fl == g.FLAG_GAS, 0.0, 1.0)
    if twin.fs_iface_flag() >= 0:
        fill = np.where(fl == g.FLAG_GAS, 0.0, np.clip(g.phi.to_numpy(), 0.0, 1.0))
    return float((fill * H).sum() * g.dx ** 3)


def _surface_source(fslbm: bool) -> None:
    from waam_twin.physics.arc import SurfaceGaussian
    twin = WAAMTwin(nx=40, ny=40, nz=16, dx=4e-4, max_tracers=4, enable_vof=True)
    twin.use_free_surface_lbm = fslbm
    twin.reset()
    g = twin.grid
    k_top = int(twin.nz_solid) - 1
    if fslbm:
        # A partly filled interface layer on top of the plate in one quadrant.
        phi = g.phi.to_numpy(); fl = g.flags.to_numpy()
        phi[:20, :20, k_top + 1] = 0.4
        fl[:20, :20, k_top + 1] = g.FLAG_IFACE
        g.phi.from_numpy(phi); g.flags.from_numpy(fl)
    twin.energy_split = False
    twin.Q_w, twin.eta = 1500.0, 0.8
    twin.sigma_cells = 1.5e-3 / g.dx
    H0 = g.H.to_numpy().astype(np.float64)
    E0 = _energy(twin)
    ai, aj = 20.0, 20.0
    SurfaceGaussian().inject(twin, g, ai, aj, float(k_top))
    dE = _energy(twin) - E0
    want = 1500.0 * 0.8 * g.dt
    rel = abs(dE - want) / want
    dH = g.H.to_numpy().astype(np.float64) - H0
    heated = np.argwhere(np.abs(dH) > 0)
    # Every heated cell is at the surface: nothing below the top metal layer
    # (the plate's top layer under a partly filled cell shares the load).
    k_min = int(heated[:, 2].min())
    assert k_min >= k_top - (0 if not fslbm else 0), k_min
    assert int(heated[:, 2].max()) <= k_top + 1
    # Horizontal Gaussian: column at 1σ from the axis ≈ e^-1/2 of the axis column
    # (full-metal columns, away from the partly filled quadrant).
    s = twin.sigma_cells
    col = lambda i, j: dH[i, j, :].sum()  # noqa: E731
    r = col(int(ai), int(aj + round(s))) / col(int(ai), int(aj))
    want_r = math.exp(-0.5 * (round(s) ** 2) / s ** 2)
    assert abs(r - want_r) < 0.02, (r, want_r)
    assert rel < 2e-4, (dE, want)
    print(f"[arc_input] surface source ({'fslbm' if fslbm else 'legacy'}): energy {dE:.4f} J vs "
          f"{want:.4f} J (rel {rel:.1e}); heated layers k={k_min}..{int(heated[:, 2].max())}; "
          f"1σ falloff {r:.3f} vs {want_r:.3f}")


def _energy_split() -> None:
    from waam_twin.physics.arc import _arc_power_W
    from waam_twin.physics.heat_source import energy_split_W
    twin = WAAMTwin(nx=8, ny=8, nz=8, dx=4e-4, max_tracers=4)
    twin.energy_split = True
    twin.droplet_transfer_mode = "pulsed"
    twin.pulse_frequency_hz = 150.0
    twin.Q_w, twin.eta = 2000.0, 0.82
    twin.wire_feed_m_s, twin.wire_diameter_m = 7.0 / 60.0, 1.0e-3  # Park: ~1 kW of droplet heat
    g = twin.grid
    split = energy_split_W(twin)
    q_drop = split["Q_droplet_W"]
    assert q_drop > 500.0, q_drop  # the check is meaningless without droplet power
    period_steps = int(round(1.0 / 150.0 / g.dt))
    # Booking is the absolute enthalpy H(T_drop)·V; the arc's share is the part
    # above ambient (q_drop is already relative to ambient).
    from waam_twin.physics.heat_source import wire_volumetric_enthalpy_J_m3
    h_drop = wire_volumetric_enthalpy_J_m3(twin, split["T_drop_K"])
    h_amb = wire_volumetric_enthalpy_J_m3(twin, twin.T_amb)
    e_drop = q_drop * period_steps * g.dt * h_drop / (h_drop - h_amb)  # absolute booking
    arc = []
    for step in range(period_steps * 400):
        twin._step_n = step
        twin._droplet_enthalpy_booked_J_prev = e_drop if (step % period_steps == 1) else 0.0
        q, eta = _arc_power_W(twin)
        assert q >= 0.0 and eta == 1.0
        arc.append(q)
    mean_arc = float(np.mean(arc[len(arc) // 2:]))
    want = split["Q_total_W"] - q_drop
    rel = abs(mean_arc - want) / want
    assert rel < 0.02, (mean_arc, want)
    print(f"[arc_input] energy split: mean arc {mean_arc:.1f} W vs total − droplets {want:.1f} W "
          f"(rel {rel:.1e}); droplets {q_drop:.0f} W of {split['Q_total_W']:.0f} W")


def _pulse_current() -> None:
    from waam_twin.job import electrical_arc_power_W, pulse_current_mean_rms_A
    from waam_twin.physics.weld_forces import force_current_A
    proc = {"pulse_peak_A": 345.0, "pulse_base_A": 35.0, "pulse_peak_ms": 1.2,
            "pulse_frequency_hz": 151.5, "voltage_V": 22.0}
    i_mean, i_rms = pulse_current_mean_rms_A(proc)
    assert abs(i_mean - 91.4) < 0.1 and abs(i_rms - 150.5) < 0.2, (i_mean, i_rms)
    # Pulsed power: U = U0 + R·I with mean voltage 22 V → V·I_mean + R(I_rms² − I_mean²).
    p_w = electrical_arc_power_W(proc, required=True)
    want_p = 22.0 * i_mean + 0.05 * (i_rms ** 2 - i_mean ** 2)
    assert abs(p_w - want_p) < 1e-6, (p_w, want_p)
    # Same as integrating U(t)·I(t) over the rectangular train.
    d = 1.2e-3 * 151.5
    u0 = 22.0 - 0.05 * i_mean
    direct = d * (u0 + 0.05 * 345.0) * 345.0 + (1 - d) * (u0 + 0.05 * 35.0) * 35.0
    assert abs(p_w - direct) < 1e-6, (p_w, direct)
    assert abs(electrical_arc_power_W(dict(proc, arc_slope_ohm=0.0)) - 22.0 * i_mean) < 1e-6
    # Steady current is unchanged.
    assert abs(electrical_arc_power_W({"current_A": 91.0, "voltage_V": 22.0}) - 2002.0) < 1e-9
    twin = WAAMTwin(nx=8, ny=8, nz=8, dx=4e-4, max_tracers=4)
    twin.welding_current_A = i_mean
    assert force_current_A(twin) == i_mean
    twin.welding_current_rms_A = i_rms
    assert force_current_A(twin) == i_rms
    print(f"[arc_input] pulse: I_mean {i_mean:.1f} A, I_rms {i_rms:.1f} A, mean U·I {p_w:.0f} W "
          f"(V·I_mean {22.0 * i_mean:.0f} W); "
          f"(arc force ×{(i_rms / i_mean) ** 2:.2f} vs a steady mean-current arc)")


def _surface_height() -> None:
    from waam_twin.compiler import ti
    from waam_twin.kernels.grid_init import surface_height_at
    GAS = 2
    for nz, top in ((64, 22), (84, 30), (128, 44), (128, 100), (256, 200)):
        phi = ti.field(ti.f32, shape=(3, 3, nz))
        flags = ti.field(ti.i32, shape=(3, 3, nz))
        out = ti.field(ti.f32, shape=())
        f = np.full((3, 3, nz), GAS, np.int32)
        p = np.zeros((3, 3, nz), np.float32)
        f[:, :, :top + 1] = 1
        p[:, :, :top + 1] = 1.0
        p[1, 1, top] = 0.04  # a near-empty top cell does not count
        flags.from_numpy(f)
        phi.from_numpy(p)
        for _ in range(10):
            surface_height_at(phi, flags, out, 1, 1, 7, GAS, nz)
            assert float(out[None]) == top - 1, (nz, top, float(out[None]))
        # All gas: fall back to nz_fallback − 1.
        flags.from_numpy(np.full((3, 3, nz), GAS, np.int32))
        surface_height_at(phi, flags, out, 1, 1, 7, GAS, nz)
        assert float(out[None]) == 6.0, float(out[None])
    print("[arc_input] surface height: top metal cell found for nz 64–256 (no warp race)")


def run() -> None:
    init_taichi()
    _surface_height()
    _surface_source(fslbm=False)
    _surface_source(fslbm=True)
    _energy_split()
    _pulse_current()


if __name__ == "__main__":
    run()
    print("PASS")
