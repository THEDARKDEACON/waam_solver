"""
test_force_additivity.py — CSF + Marangoni must accumulate (FC-1/FC-2).

PHYSICS_FORCE_CORRECTNESS_SPEC §5.1: Marangoni uses += and must not wipe CSF.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics import forces


def _interface_fixture(twin: WAAMTwin):
    """
    Soft hemispherical metal drop (κ ≠ 0) with a mild T gradient along +x.

    A flat φ=0.5 plane has Brackbill κ≈0, so CSF force vanishes — use curvature
    so the additivity check has a nonzero CSF contribution.
    """
    g = twin.grid
    twin.reset()
    T_np = g.T.to_numpy()
    phi_np = np.zeros((g.nx, g.ny, g.nz), dtype=np.float32)
    fl_np = np.zeros_like(phi_np)
    flags_np = np.full((g.nx, g.ny, g.nz), g.FLAG_GAS, dtype=np.int32)
    nz_s = twin.nz_solid
    cx = (g.nx - 1) * 0.5
    cy = (g.ny - 1) * 0.5
    # Centre just above the solid so the free surface is curved.
    cz = float(nz_s) + 3.5
    R = 5.0
    delta = 1.0
    for i in range(g.nx):
        for j in range(g.ny):
            for k in range(g.nz):
                T_np[i, j, k] = 300.0 + 250.0 * (i / max(g.nx - 1, 1))
                if k < nz_s:
                    phi_np[i, j, k] = 1.0
                    fl_np[i, j, k] = 0.0
                    flags_np[i, j, k] = g.FLAG_SOLID
                    T_np[i, j, k] = 300.0
                    continue
                r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                phi = 0.5 * (1.0 - np.tanh((r - R) / delta))
                phi_np[i, j, k] = phi
                if phi > 0.95:
                    fl_np[i, j, k] = 1.0
                    flags_np[i, j, k] = g.FLAG_FLUID
                elif phi > 0.05:
                    fl_np[i, j, k] = 1.0
                    flags_np[i, j, k] = g.FLAG_IFACE
                else:
                    fl_np[i, j, k] = 0.0
                    flags_np[i, j, k] = g.FLAG_GAS
    g.T.from_numpy(T_np)
    g.phi.from_numpy(phi_np)
    g.f_l.from_numpy(fl_np)
    g.flags.from_numpy(flags_np)
    return g


def _force_stack(g):
    return (
        g.Fx.to_numpy().copy(),
        g.Fy.to_numpy().copy(),
        g.Fz.to_numpy().copy(),
    )


def run(tol: float = 1e-9) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=32, ny=16, nz=20, dx=3e-4,
        enable_csf_tension=True,
        max_tracers=10,
    )
    g = _interface_fixture(twin)

    # --- CSF alone ---
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    Fx_csf, Fy_csf, Fz_csf = _force_stack(g)
    csf_mag = float(np.sqrt(Fx_csf**2 + Fy_csf**2 + Fz_csf**2).max())
    if csf_mag < 1e-12:
        raise AssertionError("CSF produced no force on interface fixture")

    # --- Marangoni alone ---
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_marangoni_force(
        g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
        twin.dgamma_dT_lu, g.dx,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    Fx_m, Fy_m, Fz_m = _force_stack(g)
    m_mag = float(np.sqrt(Fx_m**2 + Fy_m**2 + Fz_m**2).max())
    if m_mag < 1e-12:
        raise AssertionError("Marangoni produced no force on T-gradient interface")

    # --- CSF then Marangoni (production order) ---
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    forces.compute_marangoni_force(
        g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
        twin.dgamma_dT_lu, g.dx,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    Fx_b, Fy_b, Fz_b = _force_stack(g)

    err = max(
        float(np.max(np.abs(Fx_b - Fx_csf - Fx_m))),
        float(np.max(np.abs(Fy_b - Fy_csf - Fy_m))),
        float(np.max(np.abs(Fz_b - Fz_csf - Fz_m))),
    )
    print(
        f"[force_additivity] CSF_max={csf_mag:.3e}  Mar_max={m_mag:.3e}  "
        f"superposition_err={err:.3e}  (tol={tol:.0e})"
    )
    if err > tol:
        raise AssertionError(
            f"CSF+Marangoni not additive: max|F_both - F_csf - F_m|={err:.3e} > {tol:.0e} "
            f"(Marangoni must use +=, not overwrite)"
        )

    # Isothermal: Marangoni ~0 must leave CSF intact
    T_flat = np.full_like(g.T.to_numpy(), 1700.0)
    g.T.from_numpy(T_flat)
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    Fx_c2, Fy_c2, Fz_c2 = _force_stack(g)
    forces.compute_marangoni_force(
        g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags,
        twin.dgamma_dT_lu, g.dx,
        g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    Fx_iso, Fy_iso, Fz_iso = _force_stack(g)
    err_iso = max(
        float(np.max(np.abs(Fx_iso - Fx_c2))),
        float(np.max(np.abs(Fy_iso - Fy_c2))),
        float(np.max(np.abs(Fz_iso - Fz_c2))),
    )
    print(f"[force_additivity] isothermal CSF preserve err={err_iso:.3e}")
    if err_iso > 1e-8:
        raise AssertionError(
            f"Isothermal Marangoni altered CSF (err={err_iso:.3e}); overwrite bug?"
        )
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
