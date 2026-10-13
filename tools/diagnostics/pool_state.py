"""
pool_state.py — liquid / film / flag diagnostics for a running twin.

Usage:
  PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.diagnostics.pool_state \\
      --job jobs/examples/bead_calibrate.yaml --steps 6000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np


def diagnose(twin) -> dict:
    g = twin.grid
    nz = int(getattr(twin, "nz_solid", 0))
    fl = g.f_l.to_numpy()
    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()
    T = g.T.to_numpy()
    u = np.sqrt(
        g.ux.to_numpy() ** 2 + g.uy.to_numpy() ** 2 + g.uz.to_numpy() ** 2
    ) * g.dx / g.dt

    kk = np.arange(g.nz)[None, None, :] * np.ones_like(fl)
    liq = fl > 0.5
    sub = liq & (kk < nz)
    dep = liq & (kk >= nz)
    above = (kk >= nz) & (flags != g.FLAG_GAS)
    film = above & (flags == g.FLAG_IFACE) & (phi > 0.3) & (phi < 0.5) & (T < 400)

    dx3 = g.dx ** 3
    rho = float(twin.mat.rho)

    flag_hist = {}
    for v, c in zip(*np.unique(flags, return_counts=True)):
        flag_hist[int(v)] = int(c)

    return {
        "step": int(getattr(twin, "_step_n", 0)),
        "nz_solid": nz,
        "n_liquid_total": int(liq.sum()),
        "n_liquid_in_plate": int(sub.sum()),
        "n_liquid_above_plate": int(dep.sum()),
        "max_u_liquid_ms": float(u[liq].max()) if liq.any() else 0.0,
        "max_u_plate_liquid_ms": float(u[sub].max()) if sub.any() else 0.0,
        "n_non_gas_above_plate": int(above.sum()),
        "n_cold_iface_film": int(film.sum()),
        "film_metal_g": float(phi[film].sum() * dx3 * rho * 1e3),
        "phi_metal_above_g": float(
            (phi * above).sum() * dx3 * rho * 1e3
        ),
        "deposited_mass_g": float(
            getattr(twin, "_deposited_volume_m3", 0.0) * rho * 1e3
        ),
        "force_clamp_steps": int(getattr(twin, "_force_clamp_steps", 0)),
        "mach_clamp_steps": int(getattr(twin, "_mach_clamp_steps", 0)),
        "flag_histogram": flag_hist,
        "pool_flows": bool(liq.any() and float(u[liq].max()) > 1e-6),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", default="jobs/examples/bead_calibrate.yaml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--backend", default=os.environ.get("WAAM_BACKEND", "cuda"))
    args = p.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver

    init_taichi(backend=args.backend)
    twin = WAAMTwin.from_job(args.job)
    twin.reset()
    job = load_job_config(args.job)
    drv = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
    it = drv.positions_for_steps(args.steps, twin.grid.dt)
    for _ in range(args.steps):
        _, x, y, z = next(it)
        twin.step(x, y, True, torch_z_m=z)

    report = diagnose(twin)
    from waam_twin.measure import measure_all

    report["geometry"] = measure_all(twin)
    print(json.dumps(report, indent=2))
    return 0 if report["pool_flows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
