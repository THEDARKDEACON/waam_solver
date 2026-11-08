"""CLI arguments for the interactive viewer."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WAAM Twin v2 — interactive melt-pool viewer (PyVista)",
    )
    p.add_argument(
        "--job",
        default=None,
        help="Job YAML (default: jobs/examples/bead_calibrate.yaml, free-surface LBM)",
    )
    p.add_argument(
        "--preset",
        default=None,
        help=(
            "Hardware profile override (minimal|standard|high|ultra). "
            "Selects collision model and tracer count. Does not rewrite "
            "job domain_mm, plate.size_mm, or dx_mm."
        ),
    )
    p.add_argument(
        "--auto-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Accepted for compatibility. dx_mm is always allocated as requested."
        ),
    )
    p.add_argument(
        "--material",
        default=None,
        help="Override material path when using --preset without --job",
    )
    p.add_argument(
        "--steps-per-frame",
        type=int,
        default=int(os.environ.get("WAAM_VIEWER_SPF", "0")),
        help=(
            "LBM steps per rendered frame. 0 (default) = auto: as many steps as "
            "fit in --frame-time of physics, so the bead grows at the fastest "
            "rate the GPU allows while the window stays responsive"
        ),
    )
    p.add_argument(
        "--frame-time",
        type=float,
        default=float(os.environ.get("WAAM_VIEWER_FRAME_S", "0.3")),
        help="Physics wall time per frame in auto mode, seconds (default: 0.3)",
    )
    p.add_argument(
        "--sparse",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the job's simulation.sparse_grid (GPU only: compute only "
            "near metal; about 1.6x faster on the calibration job)"
        ),
    )
    p.add_argument(
        "--particles",
        action="store_true",
        help="Start with cell particles instead of the φ = 0.5 bead surface",
    )
    p.add_argument(
        "--paused",
        action="store_true",
        help="Start paused",
    )
    p.add_argument(
        "--liquid-only",
        action="store_true",
        help="Start with liquid-only render filter",
    )
    p.add_argument(
        "--particle-scale",
        type=float,
        default=float(os.environ.get("WAAM_VIEWER_PARTICLE_SCALE", "0.35")),
        help="Particle radius as fraction of cell size dx (default: 0.35; was 0.48)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for screenshots and VTK dumps (default: <repo>/viewer_output)",
    )
    return p
