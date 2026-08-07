"""
test_intensive_marangoni_circulation.py — Sustained thermocapillary circulation.

1. Instantaneous force-sign check (dγ/dT < 0 → hot→cold).
2. Seeded-pool on/off ablation: Marangoni-only vs forces-off must raise
   surface speed and keep f_marangoni_max > 0 over a sustained weld.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.physics import forces
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.tools.force_ablation import AblationCase, _configure, _seed_pool


def _force_sign_check() -> float:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(nx=40, ny=16, nz=20, dx=2.5e-4, max_tracers=10)
    twin.reset()
    g = twin.grid
    T_np = g.T.to_numpy()
    phi_np = g.phi.to_numpy()
    fl_np = g.f_l.to_numpy()
    flags_np = g.flags.to_numpy()
    nz_s = twin.nz_solid
    j = g.ny // 2
    for i in range(g.nx):
        for k in range(g.nz):
            if k < nz_s - 1:
                phi_np[i, j, k] = 1.0
                fl_np[i, j, k] = 0.0
                flags_np[i, j, k] = g.FLAG_SOLID
            elif k == nz_s - 1:
                phi_np[i, j, k] = 0.5
                fl_np[i, j, k] = 1.0
                flags_np[i, j, k] = g.FLAG_IFACE
                T_np[i, j, k] = 300.0 + 200.0 * (i / max(g.nx - 1, 1))
            else:
                phi_np[i, j, k] = 0.0
                fl_np[i, j, k] = 0.0
                flags_np[i, j, k] = g.FLAG_GAS
    g.T.from_numpy(T_np)
    g.phi.from_numpy(phi_np)
    g.f_l.from_numpy(fl_np)
    g.flags.from_numpy(flags_np)
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_marangoni_force(
        g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
        twin.dgamma_dT_lu, g.dx,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    Fx = g.Fx.to_numpy()[:, j, nz_s - 1]
    dT = np.gradient(T_np[:, j, nz_s - 1])
    hot_side = float(Fx[dT > 0].mean()) if np.any(dT > 0) else 0.0
    if hot_side >= 0.0:
        raise AssertionError("Marangoni force should pull hot→cold (Fx < 0 on +dT/dx)")
    return hot_side


def _run_ablation(case: AblationCase, n_steps: int) -> dict:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin(
        nx=48, ny=28, nz=28, dx=3.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        heat_source="goldak",
        arc_power_W=3200.0,
        arc_efficiency=0.75,
        welding_current_A=180.0,
        travel_speed_m_s=0.0,
        droplet_freq_hz=0.0,
        enable_recoil=False,
        max_tracers=40,
    )
    twin.wire_feed_m_s = 0.0
    twin.enable_evaporative_cooling = False
    twin.enable_gas_shear = False
    twin.enable_lorentz = False
    _configure(twin, case)
    # Isolate CSF so on/off difference is thermocapillary, not surface tension.
    twin.enable_csf_tension = False
    twin.reset()
    _seed_pool(twin, radius=6)
    g = twin.grid
    # Keep the seeded patch hot so it stays liquid for the whole soak.
    T_np = g.T.to_numpy()
    H_np = g.H.to_numpy()
    fl_np = g.f_l.to_numpy()
    hot = fl_np > 0.3
    T_keep = twin.mat.T_liquidus + 400.0
    T_np[hot] = np.maximum(T_np[hot], T_keep)
    H_np[hot] = twin.H_liq + twin.cp_rho * (T_np[hot] - twin.mat.T_liquidus)
    g.T.from_numpy(T_np)
    g.H.from_numpy(H_np)

    x = (g.nx // 3) * g.dx
    y = (g.ny // 2) * g.dx
    for _ in range(n_steps):
        twin.step(x, y, is_welding=True)

    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    ux = g.ux.to_numpy()
    uy = g.uy.to_numpy()
    fl = g.f_l.to_numpy()
    mask = fl > 0.3
    u_max = 0.0
    if mask.any():
        u_max = float(np.sqrt(ux[mask] ** 2 + uy[mask] ** 2).max()) * g.dx / g.dt
    return {
        "u_max": u_max,
        "u_telem": float(telem.get("marangoni_vel_ms", 0.0)),
        "f_ma": float(fd.get("f_marangoni_max", 0.0)),
        "n_liq": int(telem.get("n_liquid_cells", 0)),
        "T_peak": float(telem.get("peak_temp_K", 0.0)),
    }


def run(n_steps: int | None = None) -> float:
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_INTENSIVE_MA_STEPS", "600"))

    hot_side = _force_sign_check()

    on = _run_ablation(
        AblationCase(
            "marangoni_only",
            lorentz=False, buoyancy=False, droplet=False,
            gas_shear=False, arc_pressure=False,
        ),
        n_steps,
    )
    off = _run_ablation(
        AblationCase(
            "no_marangoni",
            marangoni=False, lorentz=False, buoyancy=False,
            droplet=False, gas_shear=False, arc_pressure=False,
        ),
        n_steps,
    )

    print(
        f"[intensive_marangoni] steps={n_steps}  hot_side_Fx={hot_side:.3e}  "
        f"u_on={on['u_max']:.4f} u_off={off['u_max']:.4f} m/s  "
        f"f_Ma={on['f_ma']:.2e}/{off['f_ma']:.2e}  "
        f"n_liq={on['n_liq']}/{off['n_liq']}  T={on['T_peak']:.0f}/{off['T_peak']:.0f}K"
    )
    if on["n_liq"] < 10:
        raise AssertionError(f"Marangoni-on pool frozen: n_liq={on['n_liq']}")
    if on["f_ma"] <= 0.0:
        raise AssertionError("Marangoni-on must report f_marangoni_max > 0")
    if off["f_ma"] > 1e-12:
        raise AssertionError(f"Marangoni-off must zero f_marangoni_max, got {off['f_ma']}")
    if on["u_max"] <= off["u_max"]:
        raise AssertionError(
            f"Marangoni on should raise surface speed (on={on['u_max']:.4f}, off={off['u_max']:.4f})"
        )
    if on["u_max"] < 1e-5:
        raise AssertionError(f"Marangoni circulation too weak: u={on['u_max']:.2e} m/s")
    return on["u_max"] - off["u_max"]


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
