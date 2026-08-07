#!/usr/bin/env python3
"""Headless production / HPC batch runner for waam_twin.

Examples (from the waam_twin repo root, with the package installed):

  python scripts/hpc/run_batch.py \\
    --job jobs/examples/bead_calibrate.yaml \\
    --preset high \\
    --out runs/calibrate_high

  # Cover the full torch path (recommended):
  python scripts/hpc/run_batch.py --job jobs/examples/bead_calibrate.yaml --n-steps auto

  # Fixed step count (can end mid-bead if too small):
  python scripts/hpc/run_batch.py --job jobs/examples/bead_calibrate.yaml --n-steps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    # scripts/hpc/run_batch.py → waam_twin/
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="waam_twin headless batch runner (HPC / CI)")
    p.add_argument(
        "--job",
        default=os.environ.get("WAAM_JOB", "jobs/examples/bead_calibrate.yaml"),
        help="Job YAML path (relative to repo root or absolute)",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="Hardware preset override (minimal|standard|high|ultra). "
        "Default: use simulation.preset from the job.",
    )
    p.add_argument(
        "--n-steps",
        default="auto",
        help="'auto' = cover full torch path (run_path n_steps=None). "
        "Integer = fixed step count.",
    )
    p.add_argument(
        "--out",
        default="runs/batch",
        help="Output directory for telemetry + research bundle",
    )
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip export_research_bundle (telemetry JSON only)",
    )
    p.add_argument(
        "--headless-vtk-skip",
        action="store_true",
        help="Set WAAM_HEADLESS=1 (skip VTK writers entirely)",
    )
    args = p.parse_args(argv)

    root = _repo_root()
    os.chdir(root)
    if str(root.parent) not in sys.path and str(root) not in sys.path:
        # Editable install normally makes this unnecessary; keep a fallback.
        sys.path.insert(0, str(root.parent))

    if args.headless_vtk_skip:
        os.environ["WAAM_HEADLESS"] = "1"

    os.environ.setdefault("WAAM_BACKEND", "cuda")

    from waam_twin.platform import init_taichi
    from waam_twin import WAAMTwin

    job_path = Path(args.job)
    if not job_path.is_absolute():
        job_path = root / job_path
    if not job_path.is_file():
        print(f"[run_batch] job not found: {job_path}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    n_steps: int | None
    if str(args.n_steps).lower() in ("auto", "none", ""):
        n_steps = None
    else:
        n_steps = int(args.n_steps)

    print(f"[run_batch] cwd={root}")
    print(f"[run_batch] job={job_path}")
    print(f"[run_batch] preset_override={args.preset!r}  n_steps={n_steps!r}")
    print(f"[run_batch] WAAM_BACKEND={os.environ.get('WAAM_BACKEND')}")
    print(f"[run_batch] out={out}")

    t0 = time.perf_counter()
    init_taichi()
    twin = WAAMTwin.from_job(str(job_path), preset_override=args.preset)
    twin.reset()
    twin.run_path(str(job_path), n_steps=n_steps)
    elapsed = time.perf_counter() - t0

    telem = twin.get_telemetry()
    telem["_batch"] = {
        "job": str(job_path),
        "preset_override": args.preset,
        "n_steps_arg": args.n_steps,
        "steps_executed": int(getattr(twin, "_step_n", 0)),
        "dt_s": float(twin.grid.dt),
        "dx_mm": float(twin.grid.dx) * 1000.0,
        "grid": [int(twin.grid.nx), int(twin.grid.ny), int(twin.grid.nz)],
        "wall_s": elapsed,
    }
    telem_path = out / "telemetry.json"
    telem_path.write_text(json.dumps(telem, indent=2, default=str))
    print(f"[run_batch] wrote {telem_path}")
    print(
        f"[run_batch] steps={telem['_batch']['steps_executed']}  "
        f"dt={telem['_batch']['dt_s']*1e6:.2f}µs  "
        f"dx={telem['_batch']['dx_mm']:.3f}mm  "
        f"grid={telem['_batch']['grid']}  "
        f"wall={elapsed:.1f}s"
    )

    if not args.no_bundle and os.environ.get("WAAM_HEADLESS") != "1":
        bundle_dir = out / "bundle"
        twin.export_research_bundle(str(bundle_dir))
        print(f"[run_batch] wrote research bundle → {bundle_dir}")
    elif args.no_bundle:
        print("[run_batch] skipped bundle (--no-bundle)")
    else:
        print("[run_batch] skipped bundle (WAAM_HEADLESS=1)")

    print("[run_batch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
