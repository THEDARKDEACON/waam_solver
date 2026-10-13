"""
Phase 9 — execute Ansys comparison from the export package.

Usage:
  PYTHONPATH=.. python -m waam_twin.tools.ansys_compare_run \\
      --export runs/ansys_export_bundle --ref validation/baselines/ansys_ref.json

Writes a decision rubric JSON. Does not invent Ansys results — fails clearly
if the reference file is missing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="Directory from ansys_compare_export")
    ap.add_argument("--ref", required=True, help="Ansys reference JSON (W/D/…)")
    ap.add_argument("--out", default="validation/baselines/ansys_compare_latest.json")
    ap.add_argument("--gate-pct", type=float, default=15.0)
    args = ap.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[1]
    ref_path = pathlib.Path(args.ref)
    if not ref_path.is_absolute():
        ref_path = root / ref_path
    if not ref_path.exists():
        report = {
            "status": "PENDING",
            "reason": f"Ansys reference missing: {ref_path}",
            "note": (
                "Scaffolding only — Ansys 2024 R2 comparison has not been executed. "
                "Populate --ref with measured Ansys W/D before claiming cross-code agreement."
            ),
        }
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    ref = json.loads(ref_path.read_text())
    exp_dir = pathlib.Path(args.export)
    twin_metrics_path = exp_dir / "twin_metrics.json"
    if not twin_metrics_path.exists():
        print(f"missing {twin_metrics_path}", file=sys.stderr)
        return 2
    twin = json.loads(twin_metrics_path.read_text())

    def err(key: str) -> float | None:
        if key not in ref or key not in twin:
            return None
        return 100.0 * abs(float(twin[key]) - float(ref[key])) / max(abs(float(ref[key])), 1e-9)

    errs = {
        "pool_width_pct": err("pool_width_mm"),
        "pool_depth_pct": err("pool_depth_mm"),
    }
    ok = all(v is not None and v <= args.gate_pct for v in errs.values())
    report = {
        "status": "PASS" if ok else "FAIL",
        "gate_pct": args.gate_pct,
        "errors": errs,
        "twin": twin,
        "ref": ref,
    }
    out = root / args.out
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
