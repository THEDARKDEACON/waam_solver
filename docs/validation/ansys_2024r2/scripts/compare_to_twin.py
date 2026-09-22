#!/usr/bin/env python3
"""Compare Ansys Tier A metrics JSON to waam_twin metrics JSON.

Usage:
  python3 compare_to_twin.py \\
    --ansys ../comparison/ansys_metrics.json \\
    --twin  ../comparison/twin_metrics.json \\
    --out   ../comparison/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _macro(d: dict) -> dict:
    return d.get("macro") or {}


def rel_err(a: float, b: float) -> float | None:
    if b == 0 and a == 0:
        return 0.0
    if b == 0:
        return None
    return abs(a - b) / abs(b) * 100.0


def fmt(err: float | None) -> str:
    return "n/a" if err is None else f"{err:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ansys", type=Path, required=True)
    ap.add_argument("--twin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol_pct", type=float, default=15.0, help="W/D agree band")
    args = ap.parse_args()

    ansys = json.loads(args.ansys.read_text(encoding="utf-8"))
    twin = json.loads(args.twin.read_text(encoding="utf-8"))
    am, tm = _macro(ansys), _macro(twin)

    Wa, Da = float(am.get("pool_width_mm", 0)), float(am.get("pool_depth_mm", 0))
    Wt, Dt = float(tm.get("pool_width_mm", 0)), float(tm.get("pool_depth_mm", 0))
    eW, eD = rel_err(Wa, Wt), rel_err(Da, Dt)

    lines = [
        "# Ansys vs waam_twin comparison",
        "",
        f"- Ansys: `{args.ansys}` ({ansys.get('tier', '?')})",
        f"- Twin: `{args.twin}`",
        f"- Tolerance band: ±{args.tol_pct:.0f}% on W and D",
        "",
        "| Metric | Ansys | Twin | Rel. err % |",
        "|--------|------:|-----:|-----------:|",
        f"| Pool width mm | {Wa:.3f} | {Wt:.3f} | {fmt(eW)} |",
        f"| Pool depth mm | {Da:.3f} | {Dt:.3f} | {fmt(eD)} |",
        "",
    ]

    ok_w = eW is not None and eW <= args.tol_pct
    ok_d = eD is not None and eD <= args.tol_pct
    if ok_w and ok_d:
        verdict = (
            "**Agree within band** on isotherm/pool W and D. "
            "Commercial thermal FEM may be enough for macro W/D; "
            "pursue waam_twin mainly for free-surface bead, multiphysics, or "
            "held-out cases where Ansys drifts."
        )
    elif Wa < 0.5 or Da < 0.2 or Wt < 0.5 or Dt < 0.2:
        verdict = (
            "**Under-developed or failed pool** on one side — fix mesh/power/time "
            "before interpreting twin necessity."
        )
    else:
        verdict = (
            "**Disagreement outside band.** Inspect: latent heat, η double-count, "
            "Goldak units, mesh, T_liq definition. If Ansys is conduction-faithful "
            "and still misses experimental penetration/toe trends, that supports "
            "continuing waam_twin (flow + free surface)."
        )

    lines += [
        "## Verdict",
        "",
        verdict,
        "",
        "## Notes",
        "",
        "- Tier A Ansys has **no bead crown** — do not score height/toe here.",
        "- Prefer twin job with **validated** ER70S-6 material for publishable compare.",
        "- See `decision_rubric.md` for go/no-go on continuing the twin.",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"W: Ansys {Wa:.2f} vs Twin {Wt:.2f} ({fmt(eW)}%)")
    print(f"D: Ansys {Da:.2f} vs Twin {Dt:.2f} ({fmt(eD)}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
