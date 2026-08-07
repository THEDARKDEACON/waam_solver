"""
test_recoil_sweep_smoke.py — C_acc scales soft-onset CC recoil force.

Direct force probe (same pattern as intensive_vapor_physics):
  - locked C_acc=0.25 yields |F| > 0 in the soft-onset band
  - higher C_acc increases |F_recoil|
  - lower C_acc decreases |F_recoil|
Full-bead W/D evidence: tools/recoil_accommodation_sweep (keep lock at 0.25 —
Knight 0.54 worsens macrograph W/D).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from waam_twin import WAAMTwin, kernels
from waam_twin.job import load_job_config
from waam_twin.physics import weld_forces
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.prediction import CALIBRATE_JOB


def _seed_hot_surface(twin: WAAMTwin, T_hot: float) -> tuple[int, int, int]:
    g = twin.grid
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    i0, j0 = g.nx // 2, g.ny // 2
    k0 = max(1, twin.nz_solid)
    for di in range(-4, 5):
        for dj in range(-4, 5):
            i, j, k = i0 + di, j0 + dj, k0
            if not (0 <= i < g.nx and 0 <= j < g.ny):
                continue
            flags[i, j, k] = g.FLAG_FLUID
            fl[i, j, k] = 1.0
            T[i, j, k] = T_hot
            H[i, j, k] = twin.H_liq + twin.cp_rho * (T_hot - twin.mat.T_liquidus)
            phi[i, j, k] = 0.7
            if k + 1 < g.nz:
                flags[i, j, k + 1] = g.FLAG_GAS
                phi[i, j, k + 1] = 0.05
    g.T.from_numpy(T)
    g.H.from_numpy(H)
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)
    return i0, j0, k0


def _recoil_force(c_acc: float) -> float:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    twin = WAAMTwin.from_job(CALIBRATE_JOB)
    twin.recoil_accommodation = float(c_acc)
    twin.enable_recoil = True
    twin.use_recoil_clausius_clapeyron = True
    twin.reset()
    T_onset = weld_forces.recoil_onset_K(twin)
    T_soft = 0.5 * (T_onset + twin.T_boiling_K)
    i0, j0, k0 = _seed_hot_surface(twin, T_soft)
    g = twin.grid
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    weld_forces.apply_recoil(twin, g, float(i0), float(j0), float(k0))
    return float(np.abs(g.Fz.to_numpy()).max())


def run() -> dict:
    job = load_job_config(CALIBRATE_JOB)
    locked = float((job.get("advanced_physics") or {}).get("recoil_accommodation", 0.25))
    if abs(locked - 0.25) > 1e-9:
        raise AssertionError(f"calibrate C_acc lock drifted: {locked}")

    f_low = _recoil_force(0.10)
    f_mid = _recoil_force(locked)
    f_high = _recoil_force(min(0.54, locked + 0.29))

    print(
        f"[recoil_sweep] soft-onset |Fz|  "
        f"C=0.10 → {f_low:.3e}  C={locked:.2f} → {f_mid:.3e}  "
        f"C=0.54 → {f_high:.3e}"
    )
    if f_mid <= 0.0:
        raise AssertionError("locked C_acc must produce recoil force")
    if f_high < f_mid * 1.15:
        raise AssertionError(
            f"higher C_acc should strengthen recoil ({f_mid:.3e} → {f_high:.3e})"
        )
    if f_low > f_mid * 0.95:
        raise AssertionError(
            f"lower C_acc should weaken recoil ({f_low:.3e} vs locked {f_mid:.3e})"
        )
    # Ratio should roughly track C_acc (soft-onset ramp identical).
    ratio = f_mid / max(f_low, 1e-30)
    expected = locked / 0.10
    if abs(ratio - expected) / expected > 0.35:
        raise AssertionError(
            f"C_acc scaling off: |F| ratio {ratio:.2f} vs C_acc ratio {expected:.2f}"
        )
    return {"f_low": f_low, "f_locked": f_mid, "f_high": f_high}


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
