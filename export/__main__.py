"""CLI: run simulation with scheduled research VTK export."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from waam_twin.platform import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.export import export_research_sequence
    from waam_twin.export.probes import ProbeRecorder
    from waam_twin.paths import resolve_project_path

    p = argparse.ArgumentParser(description="WAAM research VTK sequence export")
    p.add_argument("--job", default="jobs/examples/bead_on_plate.yaml")
    p.add_argument("--preset", default=None)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--every", type=int, default=50, help="Export every N steps")
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--out", default="viewer_output/sequence_run")
    p.add_argument("--tiers", default="0,1,3", help="Comma-separated export tiers")
    args = p.parse_args(argv)

    init_taichi()
    job_path = resolve_project_path(args.job)
    twin = WAAMTwin.from_job(job_path, preset_override=args.preset)
    twin.reset()

    job = getattr(twin, "_job_config", {})
    probes_cfg = job.get("probes")
    if probes_cfg:
        twin.probe_recorder = ProbeRecorder.from_job_list(probes_cfg, twin)
    else:
        g = twin.grid
        twin.probe_recorder = ProbeRecorder()
        twin.probe_recorder.add_grid(g.nx // 3, g.ny // 2, twin.nz_solid, "substrate")

    tiers = tuple(int(x.strip()) for x in args.tiers.split(",") if x.strip())
    export_research_sequence(
        twin,
        args.out,
        n_steps=args.steps,
        every_n=args.every,
        max_frames=args.max_frames,
        tiers=tiers,
        job_path=str(job_path),
    )
    print(f"Done. Open {args.out}/sequence.pvd in ParaView.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
