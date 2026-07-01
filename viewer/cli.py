"""CLI arguments for the interactive viewer."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WAAM Twin v2 — interactive melt-pool viewer (Taichi GGUI)",
    )
    p.add_argument(
        "--job",
        default=None,
        help="Job YAML (default: jobs/examples/bead_on_plate.yaml if present)",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="Override job simulation preset (minimal, standard, high, ultra)",
    )
    p.add_argument(
        "--material",
        default=None,
        help="Override material path when using --preset without --job",
    )
    p.add_argument(
        "--steps-per-frame",
        type=int,
        default=int(os.environ.get("WAAM_VIEWER_SPF", "20")),
        help="LBM steps per rendered frame (default: 20)",
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
