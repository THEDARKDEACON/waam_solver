"""
profile_step.py — kernel profiler / throughput for a job.

Usage:
  PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.diagnostics.profile_step \\
      --job jobs/examples/bead_calibrate.yaml --warm 500 --n 200 --prof 0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", default="jobs/examples/bead_calibrate.yaml")
    p.add_argument("--warm", type=int, default=500)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--prof", type=int, default=0, help="1 = enable kernel profiler")
    p.add_argument("--backend", default=os.environ.get("WAAM_BACKEND", "cuda"))
    args = p.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from waam_twin.compiler import ti

    kw = {"arch": getattr(ti, args.backend, ti.cuda), "log_level": ti.WARN}
    if args.prof:
        kw["kernel_profiler"] = True
    ti.init(**kw)

    import waam_twin.runtime as rt

    rt._taichi_initialized = True
    rt._profile = rt.PlatformProfile(args.backend, args.backend, 4096, 14000, "standard")

    from waam_twin import WAAMTwin
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver

    twin = WAAMTwin.from_job(args.job)
    twin.reset()
    g = twin.grid
    job = load_job_config(args.job)
    drv = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
    it = drv.positions_for_steps(args.warm + args.n, g.dt)
    for _ in range(args.warm):
        _, x, y, z = next(it)
        twin.step(x, y, True, torch_z_m=z)
    ti.sync()
    if args.prof:
        ti.profiler.clear_kernel_profiler_info()
    t0 = time.perf_counter()
    for _ in range(args.n):
        _, x, y, z = next(it)
        twin.step(x, y, True, torch_z_m=z)
    ti.sync()
    el = time.perf_counter() - t0
    cells = g.nx * g.ny * g.nz
    report = {
        "grid": [g.nx, g.ny, g.nz],
        "cells": cells,
        "dx_m": g.dx,
        "dt_s": g.dt,
        "tau": g.tau,
        "steps_per_s": args.n / el,
        "ms_per_step": 1000.0 * el / args.n,
        "MLUPS": cells * args.n / el / 1e6,
        "force_clamp_steps": int(getattr(twin, "_force_clamp_steps", 0)),
        "mach_clamp_steps": int(getattr(twin, "_mach_clamp_steps", 0)),
        "lorentz_iters_last": getattr(twin, "_lorentz_iters_last", None),
    }
    print(json.dumps(report, indent=2))
    if args.prof:
        ti.profiler.print_kernel_profiler_info("count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
