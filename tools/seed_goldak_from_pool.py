"""
seed_goldak_from_pool.py — Trupiano-style first-attempt Goldak axes from pool W/D.

Goldak (1984) fixes the PDF shape; axes a,b,c_f,c_r are not determined by I·V alone.
Trupiano et al. (2022) seed geometry from measured weld-pool width/depth, then refine.
This tool provides the **analytical first attempt** (no FEA / NSGA-II loop):

  b_mm        ≈ pool_width / 2     (half-width semi-axis; twin ``b_mm``)
  c_mm        ≈ pool_depth         (penetration semi-axis; twin ``c_mm``)
  a_front_mm  ≈ k_f · b_mm
  a_rear_mm   ≈ k_r · b_mm
  ff, fr      with ff + fr = 2     (Goldak continuity)

Default k_f=0.75, k_r=1.5 (rear elongated) — edit after a coupon lock.

Usage:
  python3 -m waam_twin.tools.seed_goldak_from_pool --width 7 --depth 3
  python3 -m waam_twin.tools.seed_goldak_from_pool --width 7 --depth 3 --write-yaml /tmp/goldak.yaml
  python3 -m waam_twin.tools.seed_goldak_from_pool --job jobs/examples/bead_calibrate.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from waam_twin.job import load_job_config
from waam_twin.physics.arc import normalize_goldak_fractions


def seed_goldak_axes(
    pool_width_mm: float,
    pool_depth_mm: float,
    *,
    k_front: float = 0.75,
    k_rear: float = 1.5,
    ff: float = 0.6,
    fr: float = 1.4,
) -> dict[str, float]:
    """Return job-ready ``goldak:`` fields from measured pool W×D (mm)."""
    if pool_width_mm <= 0 or pool_depth_mm <= 0:
        raise ValueError("pool width and depth must be positive")
    b = 0.5 * float(pool_width_mm)
    c = float(pool_depth_mm)
    a_f = max(0.5, float(k_front) * b)
    a_r = max(a_f, float(k_rear) * b)
    ff, fr = normalize_goldak_fractions(ff, fr)
    return {
        "ff": round(ff, 4),
        "fr": round(fr, 4),
        "a_front_mm": round(a_f, 3),
        "a_rear_mm": round(a_r, 3),
        "b_mm": round(b, 3),
        "c_mm": round(c, 3),
    }


def format_yaml_block(goldak: dict[str, float]) -> str:
    lines = ["goldak:"]
    for k, v in goldak.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=float, default=None, help="Pool / fusion width mm")
    ap.add_argument("--depth", type=float, default=None, help="Pool / penetration depth mm")
    ap.add_argument("--job", type=str, default="", help="Read reference W/D from job YAML")
    ap.add_argument("--k-front", type=float, default=0.75)
    ap.add_argument("--k-rear", type=float, default=1.5)
    ap.add_argument("--ff", type=float, default=0.6)
    ap.add_argument("--fr", type=float, default=1.4)
    ap.add_argument("--write-yaml", type=str, default="", help="Write goldak block to path")
    args = ap.parse_args(argv)

    W = args.width
    D = args.depth
    if args.job:
        job = load_job_config(args.job)
        ref = job.get("reference") or {}
        W = float(ref.get("pool_width_mm") or ref.get("bead_width_mm") or W or 0)
        D = float(ref.get("pool_depth_mm") or ref.get("remelt_depth_mm") or D or 0)
    if not W or not D:
        print("Need --width/--depth or a --job with reference pool_width_mm & pool_depth_mm", file=sys.stderr)
        return 2

    g = seed_goldak_axes(W, D, k_front=args.k_front, k_rear=args.k_rear, ff=args.ff, fr=args.fr)
    print(f"# Seeded from pool W={W:.2f} × D={D:.2f} mm (Trupiano-style first attempt)")
    print(f"# Compare to locked calibrate: a_f=2.2 a_r=4.2 b=3.0 c=1.5 (W≈7 D≈3 → seed b=3.5 c=3.0)")
    print(format_yaml_block(g), end="")
    if args.write_yaml:
        path = Path(args.write_yaml)
        path.write_text(format_yaml_block(g), encoding="utf-8")
        print(f"# Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
