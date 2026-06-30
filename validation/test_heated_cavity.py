"""
test_heated_cavity.py — Differentially heated cavity: heat flows hot → cold.

Fluid domain with hot left slab; centre temperature rises after diffusion.
Uses lattice timestep dt=1.0 (same convention as test_thermal_diffusion).
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels


def run(n_steps: int = 200, min_delta_K: float = 3.0) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=20, ny=12, nz=12, dx=3e-4, max_tracers=10)
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    T_np = g.T.to_numpy()
    for i in range(g.nx):
        for j in range(g.ny):
            for k in range(g.nz):
                T_np[i, j, k] = 450.0 if i < g.nx // 3 else 300.0
    g.T.from_numpy(T_np)
    g.H.from_numpy((twin.cp_rho * T_np).astype("float32"))

    i_probe = g.nx // 3 + 1
    j_probe = g.ny // 2
    k_probe = g.nz // 2
    T0 = float(T_np[i_probe, j_probe, k_probe])

    for _ in range(n_steps):
        kernels.advect_diffuse_temperature(
            g.H, g.T, g.ux, g.uy, g.uz, g.flags,
            twin.alpha_lu, 1.0,
            g.FLAG_SOLID, g.FLAG_GAS, twin.cp_rho,
            g.nx, g.ny, g.nz,
        )
        kernels.sync_T_from_H(g.H, g.T, twin.cp_rho)

    T1 = float(g.T.to_numpy()[i_probe, j_probe, k_probe])
    delta = T1 - T0
    print(f"[heated_cavity] centre T {T0:.1f}→{T1:.1f}K  Δ={delta:.1f}K  (min {min_delta_K}K)")
    if delta < min_delta_K:
        raise AssertionError(f"Heat did not diffuse toward centre: Δ={delta:.1f}K")
    return delta


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
