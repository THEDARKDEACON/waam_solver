"""
Short equal-travel clamp check for a job.

Usage:
  PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.diagnostics.clamp_check \\
      --job jobs/examples/bead_clamp_clear.yaml --travel-mm 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--travel-mm", type=float, default=2.0)
    p.add_argument("--backend", default=os.environ.get("WAAM_BACKEND", "cuda"))
    args = p.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from waam_twin.runtime import init_taichi, reset_taichi
    from waam_twin import WAAMTwin
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver
    from waam_twin.validation.bead_helpers import steps_for_travel
    from waam_twin.benchmark import measure_fusion_zone_mm

    reset_taichi()
    init_taichi(backend=args.backend)
    twin = WAAMTwin.from_job(args.job)
    twin.strict_mode = False
    twin.warn_on_force_clamp = False
    twin.reset()
    g = twin.grid
    n_steps = steps_for_travel(
        args.travel_mm * 1e-3, twin.travel_speed_m_s, g.dt
    )
    job = load_job_config(args.job)
    drv = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
    it = drv.positions_for_steps(n_steps, g.dt)
    for _ in range(n_steps):
        _, x, y, z = next(it)
        twin.step(x, y, True, torch_z_m=z)

    tel = twin.get_telemetry()
    fz = measure_fusion_zone_mm(twin)
    out = {
        "job": args.job,
        "grid": [g.nx, g.ny, g.nz],
        "dx_mm": g.dx * 1e3,
        "dt_us": g.dt * 1e6,
        "tau": g.tau,
        "use_srt": twin.use_srt,
        "auto_dt_ma": twin.auto_dt_ma,
        "u_mach_limit_lu": twin.u_mach_limit_lu,
        "u_design_m_s": twin.u_design_m_s,
        "n_steps": n_steps,
        "travel_mm": args.travel_mm,
        "pool_width_mm": fz.get("fusion_width_mm", 0.0),
        "pool_depth_mm": fz.get("fusion_depth_mm", 0.0),
        "n_liquid": int(tel.get("n_liquid_cells", 0)),
        "peak_temp_C": float(tel.get("peak_temp_C", 0)),
        "max_u_ms": float(tel.get("marangoni_vel_ms", 0)),
        "force_clamp_steps": int(tel.get("force_clamp_steps", 0)),
        "mach_clamp_steps": int(tel.get("mach_clamp_steps", 0)),
        "clamps_clear": (
            int(tel.get("force_clamp_steps", 0)) == 0
            and int(tel.get("mach_clamp_steps", 0)) == 0
        ),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["clamps_clear"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
