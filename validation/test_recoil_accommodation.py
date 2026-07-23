"""
test_recoil_accommodation.py — CC recoil uses C_acc; zero below T_boil.
"""

from __future__ import annotations

import sys

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin import kernels


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
    g = twin.grid

    T_np = g.T.to_numpy()
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    i, j, k = g.nx // 2, g.ny // 2, twin.nz_solid + 2
    # Interface with vertical φ gradient
    phi_np[i, j, k] = 0.6
    phi_np[i, j, k + 1] = 0.2
    flags_np[i, j, k] = g.FLAG_IFACE
    flags_np[i, j, k + 1] = g.FLAG_IFACE

    T_np[...] = 2000.0  # below boil
    T_np[i, j, k] = 2000.0
    g.T.from_numpy(T_np)
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    kernels.apply_vapor_recoil_clausius_clapeyron(
        g.Fz, g.T, g.phi, g.flags,
        float(i), float(j), float(k), twin.sigma_cells,
        twin.P_vapor_ref_Pa, twin.T_boiling_K, twin.L_vapor_J_kg, twin.R_spec_vapor_J_kgK,
        twin.recoil_accommodation,
        g.dt, g.dx, twin.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    F_cold = float(g.Fz.to_numpy()[i, j, k])
    if abs(F_cold) > 1e-18:
        raise AssertionError(f"recoil should be 0 below T_boil, got Fz={F_cold}")

    T_np[i, j, k] = 3400.0
    g.T.from_numpy(T_np)
    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    kernels.apply_vapor_recoil_clausius_clapeyron(
        g.Fz, g.T, g.phi, g.flags,
        float(i), float(j), float(k), twin.sigma_cells,
        twin.P_vapor_ref_Pa, twin.T_boiling_K, twin.L_vapor_J_kg, twin.R_spec_vapor_J_kgK,
        1.0,
        g.dt, g.dx, twin.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    F_full = float(g.Fz.to_numpy()[i, j, k])

    kernels.clear_forces(g.Fx, g.Fy, g.Fz)
    kernels.apply_vapor_recoil_clausius_clapeyron(
        g.Fz, g.T, g.phi, g.flags,
        float(i), float(j), float(k), twin.sigma_cells,
        twin.P_vapor_ref_Pa, twin.T_boiling_K, twin.L_vapor_J_kg, twin.R_spec_vapor_J_kgK,
        0.54,
        g.dt, g.dx, twin.mat.rho,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    F_acc = float(g.Fz.to_numpy()[i, j, k])

    print(f"[recoil] F_cold={F_cold:.3e}  F_C=1={F_full:.3e}  F_C=0.54={F_acc:.3e}")
    if F_full >= 0.0:
        raise AssertionError("recoil above boil should push downward (Fz < 0)")
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
