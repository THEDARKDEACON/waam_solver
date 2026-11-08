"""
test_marangoni_gas_stencil.py — Marangoni ∇T must not read gas temperatures.

A tilted staircase interface with fractional fill (metal below the plane
x + 2z = c, gas above) and isothermal metal and cold gas has no surface temperature gradient, so the
Marangoni force must vanish. The central-difference ∇T used to include the
gas cells; where φ is fractional, T is not a function of φ across the stencil,
so the metal→gas jump has a tangential part that (I − n̂n̂) does not remove, which drove a spurious surface
flow from the gas temperature. Reference: the same interface with a real
tangential gradient in the metal (T = T0 + G·y).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin import kernels
from waam_twin.runtime import init_taichi

T_METAL = 2000.0
T_GAS = 300.0
G = 5.0  # K per cell along y (tangential to the plane) for the reference


def _force(twin: WAAMTwin, T_np: np.ndarray, variable: bool) -> float:
    g = twin.grid
    g.T.from_numpy(T_np.astype(np.float32))
    g.Fx.fill(0.0)
    g.Fy.fill(0.0)
    g.Fz.fill(0.0)
    if variable:
        g.dgamma_lu_field.fill(-1e-4)
        kernels.compute_marangoni_force_variable(
            g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags, g.dgamma_lu_field,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    else:
        kernels.compute_marangoni_force(
            g.T, g.phi, g.f_l, g.Fx, g.Fy, g.Fz, g.flags, -1e-4, g.dx,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    F = np.sqrt(g.Fx.to_numpy() ** 2 + g.Fy.to_numpy() ** 2 + g.Fz.to_numpy() ** 2)
    return float(F.sum())


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=20, ny=12, nz=20, dx=3e-4, enable_vof=True, max_tracers=10)
    twin.reset(test_fluid_domain=True)
    g = twin.grid
    I, J, K = np.meshgrid(np.arange(g.nx), np.arange(g.ny), np.arange(g.nz), indexing="ij")
    # Plane x + 2z = 28 (signed distance d, metal where d < 0) with a
    # resolved interface: fill = clip(0.5 − d, 0, 1) as in FSLBM IFACE cells.
    d = ((I + 0.5) + 2.0 * (K + 0.5) - 28.0) / np.sqrt(5.0)
    fill = np.clip(0.5 - d, 0.0, 1.0)
    metal = fill > 0.0
    flags = np.where(metal, g.FLAG_FLUID, g.FLAG_GAS).astype(np.int32)
    g.flags.from_numpy(flags)
    g.phi.from_numpy(fill.astype(np.float32))
    g.f_l.from_numpy(metal.astype(np.float32))

    worst = 0.0
    for variable in (False, True):
        label = "variable" if variable else "scalar"
        T_iso = np.where(metal, T_METAL, T_GAS)
        T_ref = np.where(metal, T_METAL + G * J, T_GAS)
        F_spur = _force(twin, T_iso, variable)
        F_ref = _force(twin, T_ref, variable)
        ratio = F_spur / max(F_ref, 1e-30)
        worst = max(worst, ratio)
        print(
            f"[marangoni_gas] {label:<8} isothermal metal ΣF={F_spur:.3e}  "
            f"real ∂T/∂y={G} K/cell ΣF={F_ref:.3e}  ratio={ratio:.2e}"
        )
        if F_ref <= 0.0:
            raise AssertionError(f"{label}: no Marangoni force for a real surface gradient")
        if ratio > 1e-3:
            raise AssertionError(
                f"{label}: isothermal metal gives Marangoni force {ratio:.2e} of a "
                f"{G} K/cell gradient (gas temperature leaks into ∇T)"
            )
    print("[marangoni_gas] PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
