"""
test_intensive_coupled_physics.py — Long seeded-pool stress of the full force set.

Intensive relative to force_ablation / direction smoke:
  - physics_tier full + recoil + evaporative cooling + gas shear + Lorentz
  - ≥800 coupled weld steps on a seeded pool
  - every catalogue force diagnostic must be active
  - fields finite; Ma velocity bounded; Lorentz streak controlled
"""

from __future__ import annotations

import os
import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.job import apply_physics_tier
from waam_twin.runtime import init_taichi
from waam_twin.tools.force_ablation import _seed_pool


def run(n_steps: int | None = None) -> dict:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_INTENSIVE_STEPS", "1200"))

    twin = WAAMTwin(
        nx=48, ny=28, nz=28, dx=3.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        heat_source="goldak",
        arc_power_W=3200.0,
        arc_efficiency=0.75,
        welding_current_A=180.0,
        travel_speed_m_s=0.005,
        droplet_freq_hz=40.0,
        enable_recoil=True,
        use_recoil_clausius_clapeyron=True,
        enable_evaporative_cooling=True,
        enable_enthalpy_cap=True,
        T_vapor_cap_K=3200.0,
        T_boiling_K=3000.0,
        lorentz_jacobi_iters=200,
        max_tracers=80,
    )
    twin.wire_feed_m_s = 4.0 / 60.0
    twin.recoil_accommodation = 0.25
    twin.evap_cooling_scale = 25.0
    twin.lorentz_jacobi_tol = 5.0e-4
    twin.gas_jet_velocity_m_s = 12.0
    twin.gas_shear_coeff = 1.0
    apply_physics_tier(twin, "full")
    twin.enable_recoil = True
    twin.enable_lorentz = True
    twin.enable_gas_shear = True
    twin.enable_evaporative_cooling = True
    twin.enable_force_diagnostics = True
    twin.grid.ensure_lorentz_fields()
    twin.reset()
    _seed_pool(twin, radius=6)
    # Preheat surface into soft-onset recoil band so f_recoil is measurable.
    g = twin.grid
    T_np = g.T.to_numpy()
    H_np = g.H.to_numpy()
    fl_np = g.f_l.to_numpy()
    onset = max(0.85 * twin.T_boiling_K, twin.mat.T_liquidus + 200.0)
    T_seed = 0.5 * (onset + twin.T_boiling_K)
    hot = fl_np > 0.3
    T_np[hot] = np.maximum(T_np[hot], T_seed)
    H_np[hot] = twin.H_liq + twin.cp_rho * (T_np[hot] - twin.mat.T_liquidus)
    g.T.from_numpy(T_np)
    g.H.from_numpy(H_np)

    x = (g.nx // 3) * g.dx
    y = (g.ny // 2) * g.dx
    for step in range(n_steps):
        twin.step(x, y, is_welding=True)
        x += twin.travel_speed_m_s * g.dt
        if step % 200 == 199:
            T = g.T.to_numpy()
            if not np.isfinite(T).all():
                raise AssertionError(f"NaN/Inf in T at step {step + 1}")

    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    required = (
        "f_csf_max", "f_marangoni_max", "f_arc_max", "f_lorentz_max",
        "f_gas_shear_max", "f_buoyancy_max",
    )
    missing = [k for k in required if float(fd.get(k, 0.0)) <= 0.0]
    if float(fd.get("f_recoil_max", 0.0)) <= 0.0 and float(telem.get("peak_temp_K", 0)) > onset:
        missing.append("f_recoil_max")
    u_ma = float(telem.get("marangoni_vel_ms", 0.0))
    n_liq = int(telem.get("n_liquid_cells", 0))
    streak = int(telem.get("lorentz_unconverged_streak", 0))
    peak_C = float(telem.get("peak_temp_C", 0.0))

    ux = g.ux.to_numpy()
    uy = g.uy.to_numpy()
    uz = g.uz.to_numpy()
    if not (np.isfinite(ux).all() and np.isfinite(uy).all() and np.isfinite(uz).all()):
        raise AssertionError("NaN/Inf in velocity after intensive coupled run")

    print(
        f"[intensive_coupled] steps={n_steps}  n_liq={n_liq}  T_peak={peak_C:.0f}C  "
        f"u_Ma={u_ma:.3f} m/s  Lz_streak={streak}  "
        f"Ma={fd.get('f_marangoni_max', 0):.2e}  "
        f"Lz={fd.get('f_lorentz_max', 0):.2e}  "
        f"recoil={fd.get('f_recoil_max', 0):.2e}  "
        f"arc={fd.get('f_arc_max', 0):.2e}  "
        f"shear={fd.get('f_gas_shear_max', 0):.2e}"
    )
    if n_liq < 20:
        raise AssertionError(f"intensive coupled pool too small: n_liq={n_liq}")
    if missing:
        raise AssertionError(f"inactive force diagnostics: {missing}")
    if u_ma > 5.0:
        raise AssertionError(f"Marangoni velocity unphysically large: {u_ma:.2f} m/s")
    if streak > 25:
        raise AssertionError(f"Lorentz unconverged streak {streak} > 25")
    if telem.get("vapor_cap_saturated") and peak_C > 2920:
        # Cap hit hard — intensive Goldak should prefer evaporative sink.
        raise AssertionError("vapor cap saturated under intensive coupled physics")
    return {"n_liquid": n_liq, "force_diagnostics": fd, "telem": telem}


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
