#!/usr/bin/env python3
"""Headless production / HPC batch runner for waam_twin.

Writes:
  - telemetry.json
  - bundle/          — final VTK research snapshot (.vti/.vtp) + final.pvd
  - sequence/        — time-series frames + sequence.pvd (ParaView playable)

Examples (from the waam_twin repo root, with the package installed):

  python scripts/hpc/run_batch.py \\
    --job jobs/examples/bead_on_plate_hires.yaml \\
    --n-steps auto \\
    --out runs/bead_on_plate_hires

  # Fewer ParaView frames (faster / less disk):
  python scripts/hpc/run_batch.py --job jobs/examples/bead_on_plate_hires.yaml \\
    --sequence-every 2000 --out runs/bead_on_plate_hires

  # Final VTK only (no time series):
  python scripts/hpc/run_batch.py --job jobs/examples/bead_on_plate_hires.yaml \\
    --sequence-every 0 --out runs/bead_on_plate_hires
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    # scripts/hpc/run_batch.py → waam_twin/
    return Path(__file__).resolve().parents[2]


def _run_path_with_exports(
    twin,
    job_path: Path,
    n_steps: int | None,
    *,
    out: Path,
    sequence_every: int,
    max_frames: int,
    export_vtk: bool,
) -> dict:
    """Follow torch path; optionally dump VTK frames + PVD along the way."""
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver, clamp_torch_to_domain
    from waam_twin.export.bundle import export_research_bundle, write_pvd

    job = load_job_config(job_path)
    waypoints = parse_torch_path(job)
    g = twin.grid
    if not waypoints:
        cy = (g.ny // 2) * g.dx
        waypoints = [(0.01, cy, 0.0)]
    driver = TorchPathDriver(waypoints, twin.travel_speed_m_s)

    if n_steps is None:
        n_steps = max(400, int(driver.total_length / max(twin.travel_speed_m_s * g.dt, 1e-12)))

    interpass = int(getattr(twin, "_interpass_cooling_steps", 0) or 0)
    prev_seg = -1

    if driver.segments:
        s0 = driver.segments[0]
        dxs, dys, dzs = s0.x1 - s0.x0, s0.y1 - s0.y0, s0.z1 - s0.z0
        norm = math.sqrt(dxs * dxs + dys * dys + dzs * dzs)
        if norm > 1e-12:
            twin._torch_dir_xyz = (dxs / norm, dys / norm, dzs / norm)

    def _clamp(x_m: float, y_m: float) -> tuple[float, float]:
        ox, oy, _ = twin.window_offset_m()
        cx, cy = clamp_torch_to_domain(x_m - ox, y_m - oy, g.nx, g.ny, g.dx)
        return cx + ox, cy + oy

    seq_dir = out / "sequence"
    vti_paths: list[str] = []
    frame_times_s: list[float] = []
    frame = 0
    do_seq = export_vtk and sequence_every > 0

    if do_seq:
        seq_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[run_batch] sequence export every {sequence_every} steps "
            f"(max_frames={max_frames}) → {seq_dir}"
        )

    def _export_frame() -> None:
        nonlocal frame
        if not do_seq or frame >= max_frames:
            return
        step_n = int(twin._step_n)
        tag = f"step_{step_n:06d}"
        sub = seq_dir / f"frame_{frame:04d}"
        paths = export_research_bundle(
            twin,
            sub,
            tag=tag,
            tiers=(0, 1, 3),
            include_surface=True,
            include_tracers=(frame == 0 or frame % 5 == 0),
            job_path=str(job_path),
            allow_skip=not export_vtk,
        )
        if "volume" in paths:
            vti_paths.append(paths["volume"])
            frame_times_s.append(step_n * g.dt)
            frame += 1
            print(f"[run_batch] sequence frame {frame}/{max_frames} @ step {step_n}")

    print(f"[run_batch] starting path: {n_steps} steps, path={driver.total_length*1e3:.1f} mm")
    for step, x, y, z in driver.positions_for_steps(n_steps, g.dt):
        seg = driver.segment_index_at_distance(driver.distance_at_step(step, g.dt))
        if interpass > 0 and seg > prev_seg and prev_seg >= 0:
            park = driver.segment_end(prev_seg)
            px, py, pz = park if park else (x, y, z)
            twin._idle_interpass((px, py, pz), (x, y, z), interpass, _clamp)
        prev_seg = seg
        cx, cy = _clamp(x, y)
        twin.step(cx, cy, is_welding=True, torch_z_m=z)

        step_n = int(twin._step_n)
        if do_seq and frame < max_frames:
            if step_n == 1 or (step_n % sequence_every == 0):
                _export_frame()

    # Always capture the final state in the sequence (if enabled and room left)
    if do_seq and frame < max_frames:
        last_step = int(twin._step_n)
        already = bool(vti_paths and f"step_{last_step:06d}" in Path(vti_paths[-1]).name)
        if not already:
            _export_frame()

    pvd_path = None
    if vti_paths:
        pvd_path = seq_dir / "sequence.pvd"
        write_pvd(pvd_path, vti_paths, times_s=frame_times_s)
        print(f"[run_batch] ParaView: open {pvd_path}")

    return {
        "n_steps": n_steps,
        "frames": frame,
        "sequence_pvd": str(pvd_path) if pvd_path else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="waam_twin headless batch runner (HPC)")
    p.add_argument(
        "--job",
        default=os.environ.get("WAAM_JOB", "jobs/examples/bead_on_plate_hires.yaml"),
        help="Job YAML path (relative to repo root or absolute)",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="Hardware preset override (minimal|standard|high|ultra). "
        "Default: use simulation.preset from the job.",
    )
    p.add_argument(
        "--auto-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accepted for compatibility. dx_mm is always allocated as requested.",
    )
    p.add_argument(
        "--n-steps",
        default="auto",
        help="'auto' = cover full torch path. Integer = fixed step count.",
    )
    p.add_argument(
        "--out",
        default="runs/batch",
        help="Output directory for telemetry + VTK/PVD",
    )
    p.add_argument(
        "--sequence-every",
        type=int,
        default=int(os.environ.get("WAAM_SEQUENCE_EVERY", "500")),
        help="Export a VTK frame every N steps and write sequence.pvd "
        "(0 = final snapshot only). Default: 500.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=int(os.environ.get("WAAM_MAX_FRAMES", "200")),
        help="Cap on sequence frames (default 200).",
    )
    p.add_argument(
        "--no-bundle",
        action="store_true",
        help="Skip final research bundle under out/bundle/",
    )
    p.add_argument(
        "--no-vtk",
        action="store_true",
        help="Skip VTK writers (telemetry still written). Implies allow_skip.",
    )
    p.add_argument(
        "--headless-vtk-skip",
        action="store_true",
        help="Set WAAM_HEADLESS=1 and skip VTK (same as --no-vtk)",
    )
    args = p.parse_args(argv)

    root = _repo_root()
    os.chdir(root)
    if str(root.parent) not in sys.path and str(root) not in sys.path:
        sys.path.insert(0, str(root.parent))

    if args.headless_vtk_skip or args.no_vtk:
        os.environ["WAAM_HEADLESS"] = "1"

    os.environ.setdefault("WAAM_BACKEND", "cuda")

    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.export.bundle import write_pvd
    from waam_twin.export.probes import ProbeRecorder

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

    export_vtk = os.environ.get("WAAM_HEADLESS") != "1"

    print(f"[run_batch] cwd={root}")
    print(f"[run_batch] job={job_path}")
    print(f"[run_batch] preset_override={args.preset!r}  n_steps={n_steps!r}")
    print(f"[run_batch] sequence_every={args.sequence_every}  max_frames={args.max_frames}")
    print(f"[run_batch] WAAM_BACKEND={os.environ.get('WAAM_BACKEND')}")
    print(f"[run_batch] out={out}")

    t0 = time.perf_counter()
    init_taichi()
    twin = WAAMTwin.from_job(
        str(job_path),
        preset_override=args.preset,
        auto_grid=args.auto_grid,
    )
    twin.reset()

    job_cfg = getattr(twin, "_job_config", None) or {}
    probes_cfg = job_cfg.get("probes") if isinstance(job_cfg, dict) else None
    if probes_cfg:
        twin.probe_recorder = ProbeRecorder.from_job_list(probes_cfg, twin)

    seq_info = _run_path_with_exports(
        twin,
        job_path,
        n_steps,
        out=out,
        sequence_every=args.sequence_every,
        max_frames=args.max_frames,
        export_vtk=export_vtk,
    )
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
        "sequence_frames": seq_info.get("frames"),
        "sequence_pvd": seq_info.get("sequence_pvd"),
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

    if not args.no_bundle:
        bundle_dir = out / "bundle"
        paths = twin.export_research_bundle(
            str(bundle_dir), allow_skip=not export_vtk,
        )
        print(f"[run_batch] wrote research bundle → {bundle_dir}")
        if isinstance(paths, dict) and paths.get("volume"):
            final_pvd = out / "final.pvd"
            write_pvd(final_pvd, [paths["volume"]], times_s=[twin._step_n * twin.grid.dt])
            print(f"[run_batch] ParaView (final): open {final_pvd}")
        elif not export_vtk:
            print("[run_batch] VTK writers skipped (--no-vtk / WAAM_HEADLESS=1)")
    else:
        print("[run_batch] skipped final bundle (--no-bundle)")

    if seq_info.get("sequence_pvd"):
        print(f"[run_batch] ParaView (animation): open {seq_info['sequence_pvd']}")

    print("[run_batch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
