"""
test_recoil_accommodation.py — CC recoil uses C_acc; soft onset below T_boil.
"""

from __future__ import annotations

import sys

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi
from waam_twin import kernels


def _apply(twin, g, i, j, k, C_acc: float, T_onset: float) -> float:
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    kernels.apply_vapor_recoil_clausius_clapeyron(
        g.Fz, g.T, g.phi, g.flags,
        float(i), float(j), float(k), twin.sigma_cells,
        twin.P_vapor_ref_Pa, twin.T_boiling_K, T_onset,
        twin.L_vapor_J_kg, twin.R_spec_vapor_J_kgK,
        C_acc,
        g.dt, g.dx, g.alloy_frac, twin.mat.rho, twin.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    return float(g.Fz.to_numpy()[i, j, k])


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=20, ny=16, nz=16, dx=3e-4,
        enable_recoil=True,
        use_recoil_clausius_clapeyron=True,
        max_tracers=4,
    )
    twin.reset(test_fluid_domain=True)
    twin.recoil_accommodation = 0.54
    twin.T_boiling_K = 3100.0
    T_onset = 0.85 * twin.T_boiling_K  # 2635 K
    g = twin.grid

    T_np = g.T.to_numpy()
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    i, j, k = g.nx // 2, g.ny // 2, twin.nz_solid + 2
    phi_np[i, j, k] = 0.6
    phi_np[i, j, k + 1] = 0.2
    flags_np[i, j, k] = g.FLAG_IFACE
    flags_np[i, j, k + 1] = g.FLAG_IFACE

    # Below onset → zero
    T_np[...] = 2000.0
    T_np[i, j, k] = 2000.0
    g.T.from_numpy(T_np)
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)
    F_cold = _apply(twin, g, i, j, k, 0.54, T_onset)
    if abs(F_cold) > 1e-18:
        raise AssertionError(f"recoil should be 0 below T_onset, got Fz={F_cold}")

    # Soft-onset band (between onset and boil) → nonzero, weaker than above boil
    T_np[i, j, k] = 2900.0
    g.T.from_numpy(T_np)
    F_soft = _apply(twin, g, i, j, k, 1.0, T_onset)
    if F_soft >= 0.0:
        raise AssertionError("soft-onset recoil should push downward (Fz < 0)")

    # Above boil — C_acc scaling
    T_np[i, j, k] = 3400.0
    g.T.from_numpy(T_np)
    F_full = _apply(twin, g, i, j, k, 1.0, T_onset)
    F_acc = _apply(twin, g, i, j, k, 0.54, T_onset)

    print(
        f"[recoil] F_cold={F_cold:.3e}  F_soft={F_soft:.3e}  "
        f"F_C=1={F_full:.3e}  F_C=0.54={F_acc:.3e}"
    )
    if F_full >= 0.0:
        raise AssertionError("recoil above boil should push downward (Fz < 0)")
    if abs(F_soft) >= abs(F_full):
        raise AssertionError("soft-onset |F| should be weaker than above-boil |F|")
    ratio = F_acc / F_full
    if abs(ratio - 0.54) > 0.02:
        raise AssertionError(f"C_acc scale wrong: F_0.54/F_1 = {ratio:.3f} (expect ~0.54)")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
