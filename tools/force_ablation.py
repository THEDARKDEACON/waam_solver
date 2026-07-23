"""
force_ablation.py — Cho & Na-style on/off force ranking smoke.

Runs a short fixed-torch weld with selected forces disabled and reports
pool depth, peak |u|, and force diagnostics.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cpu python -m waam_twin.tools.force_ablation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.job import apply_physics_tier
from waam_twin.platform import init_taichi


@dataclass
class AblationCase:
    name: str
    marangoni: bool = True
    lorentz: bool = True
    buoyancy: bool = True  # beta_T → 0 disables
    droplet: bool = True
    gas_shear: bool = True
    arc_pressure: bool = True


CASES = [
    AblationCase("full"),
    AblationCase("no_marangoni", marangoni=False),
    AblationCase("no_lorentz", lorentz=False),
    AblationCase("no_buoyancy", buoyancy=False),
    AblationCase("no_droplet", droplet=False),
    AblationCase("no_gas_shear", gas_shear=False),
    AblationCase("marangoni_only", lorentz=False, buoyancy=False,
                 droplet=False, gas_shear=False, arc_pressure=False),
    AblationCase("lorentz_only", marangoni=False, buoyancy=False,
                 droplet=False, gas_shear=False, arc_pressure=False),
]


def _configure(twin: WAAMTwin, case: AblationCase) -> None:
    apply_physics_tier(twin, "full")
    twin.enable_force_diagnostics = True
    twin.marangoni_scale = 1.0 if case.marangoni else 0.0
    if not case.marangoni:
        twin.dgamma_dT_lu = 0.0
    twin.enable_lorentz = case.lorentz
    if twin.enable_lorentz:
        twin.grid.ensure_lorentz_fields()
    twin.beta_T = twin.mat.beta_T if case.buoyancy else 0.0
    twin.enable_deposition_momentum = case.droplet
    twin.enable_droplet_impact_pressure = case.droplet
    if not case.droplet:
        twin.wire_feed_m_s = 0.0
        twin.droplet_freq = 0.0
    twin.enable_gas_shear = case.gas_shear
    if not case.arc_pressure:
        twin.arc_pressure_model = "constant"
        twin.arc_pressure = 0.0


def _seed_pool(twin: WAAMTwin, radius: int = 5) -> None:
    """Seed a small molten patch so force diagnostics see an interface."""
    g = twin.grid
    i0, j0 = g.nx // 3, g.ny // 2
    k0 = twin.nz_solid
    T_np = g.T.to_numpy()
    fl_np = g.f_l.to_numpy()
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    H_np = g.H.to_numpy()
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for dk in range(0, radius):
                if di * di + dj * dj + dk * dk > radius * radius:
                    continue
                i, j, k = i0 + di, j0 + dj, k0 + dk
                if not (0 <= i < g.nx and 0 <= j < g.ny and 0 <= k < g.nz):
                    continue
                if flags_np[i, j, k] == g.FLAG_SOLID and k < twin.nz_solid:
                    continue
                Tmelt = twin.mat.T_liquidus + 400.0
                T_np[i, j, k] = Tmelt
                fl_np[i, j, k] = 1.0
                phi_np[i, j, k] = 1.0
                flags_np[i, j, k] = g.FLAG_FLUID
                H_np[i, j, k] = twin.cp_rho * Tmelt + twin.L_rho
    # Soft free surface above the seed
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            i, j, k = i0 + di, j0 + dj, k0 + radius
            if 0 <= i < g.nx and 0 <= j < g.ny and 0 <= k < g.nz:
                phi_np[i, j, k] = 0.5
                fl_np[i, j, k] = 1.0
                flags_np[i, j, k] = g.FLAG_IFACE
                T_np[i, j, k] = twin.mat.T_liquidus + 200.0
                H_np[i, j, k] = twin.cp_rho * T_np[i, j, k] + twin.L_rho
    g.T.from_numpy(T_np)
    g.f_l.from_numpy(fl_np)
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)
    g.H.from_numpy(H_np)


def _run_case(case: AblationCase, n_steps: int, dx: float) -> dict[str, Any]:
    twin = WAAMTwin(
        nx=36, ny=24, nz=22, dx=dx,
        enable_vof=True,
        enable_csf_tension=True,
        heat_source="gaussian2d",
        arc_power_W=3500.0,
        arc_efficiency=0.8,
        welding_current_A=200.0,
        travel_speed_m_s=0.005,
        droplet_freq_hz=40.0,
        max_tracers=20,
        lorentz_jacobi_iters=80,
    )
    twin.wire_feed_m_s = 0.5 / 60.0
    _configure(twin, case)
    twin.reset()
    _seed_pool(twin, radius=5)
    g = twin.grid
    x0 = (g.nx // 3) * g.dx
    y0 = (g.ny // 2) * g.dx
    for _ in range(n_steps):
        twin.step(x0, y0, is_welding=True)
        x0 += twin.travel_speed_m_s * g.dt

    telem = twin.get_telemetry()
    ux = g.ux.to_numpy()
    uy = g.uy.to_numpy()
    uz = g.uz.to_numpy()
    fl = g.f_l.to_numpy()
    mask = fl > 0.5
    if mask.any():
        u_max = float(np.sqrt(ux[mask] ** 2 + uy[mask] ** 2 + uz[mask] ** 2).max()) * g.dx / g.dt
        uz_mean = float(uz[mask].mean()) * g.dx / g.dt
    else:
        u_max = uz_mean = 0.0

    return {
        "case": case.name,
        "pool_width_mm": telem["pool_width_mm"],
        "pool_depth_mm": telem["pool_depth_mm"],
        "peak_temp_K": telem["peak_temp_K"],
        "n_liquid_cells": telem["n_liquid_cells"],
        "u_max_ms": round(u_max, 5),
        "uz_mean_ms": round(uz_mean, 5),
        "force_diagnostics": telem.get("force_diagnostics", {}),
    }


def run_ablation(n_steps: int = 80, dx: float = 3.5e-4) -> list[dict[str, Any]]:
    init_taichi(backend="cpu")
    results = []
    for case in CASES:
        row = _run_case(case, n_steps=n_steps, dx=dx)
        results.append(row)
        fd = row["force_diagnostics"]
        print(
            f"[ablation] {case.name:16s}  D={row['pool_depth_mm']:.3f} mm  "
            f"W={row['pool_width_mm']:.3f} mm  u_max={row['u_max_ms']:.4f} m/s  "
            f"Ma={fd.get('f_marangoni_max', 0):.2e}  Lz={fd.get('f_lorentz_max', 0):.2e}"
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Cho & Na-style force ablation")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dx", type=float, default=3.5e-4)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()
    rows = run_ablation(n_steps=args.steps, dx=args.dx)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[ablation] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
