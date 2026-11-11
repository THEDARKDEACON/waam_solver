"""Evaluate the Lambiase held-out predictions against docs/LAMBIASE_PREREG.md.

Usage: python -m waam_twin.tools.eval_lambiase
Reads runs/calibration/sweep_lambiase.{jsonl,log} and the reference blocks of
jobs/examples/lambiase2026/runNN.yaml. Prints the per-bead table, the
pre-registered metrics (MAE, trend signs) and the Amendment-1 width check.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSONL = ROOT / "runs/calibration/sweep_lambiase.jsonl"
LOG = ROOT / "runs/calibration/sweep_lambiase.log"
JOBS = ROOT / "jobs/examples/lambiase2026"


def _ref(run: str) -> dict:
    d = yaml.safe_load((JOBS / f"{run}.yaml").read_text())
    r, p = d["reference"], d["process"]
    return {"P": p["mean_power_W"] / 1000, "v": p["travel_speed_mm_s"] * 60,
            "sod": r.get("sod_mm"), "H": r["bead_height_mm"], "CL": r["contact_length_mm"],
            "p": r["pool_depth_mm"], "Wb": r["bead_width_mm"]}


def _wbead_from_log() -> dict[str, float]:
    """W_bead of the validate #0 run of each point (last header for that point wins)."""
    out: dict[str, float] = {}
    cur = None
    for line in LOG.read_text(errors="replace").splitlines():
        m = re.match(r"=== (run\d+) ", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"RESULT .*W_bead=([\d.]+)", line)
        if m and cur:
            out[cur] = float(m.group(1))
    return out


def _slope_sign(rows, x, y):
    """Mean sign of within-group slopes of y vs x, grouping on the other factors."""
    others = [k for k in ("P", "v", "sod") if k != x]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in others), []).append(r)
    slopes = []
    for g in groups.values():
        if len(g) >= 2:
            xs = np.array([r[x] for r in g]); ys = np.array([r[y] for r in g])
            if np.ptp(xs) > 0:
                slopes.append(np.polyfit(xs, ys, 1)[0])
    return float(np.mean(slopes)) if slopes else float("nan"), len(slopes)


def main() -> None:
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    val = {r["point"]: r for r in recs if r.get("kind") == "validate" and r.get("sample", 0) == 0}
    wb = _wbead_from_log()
    rows = []
    for run, r in sorted(val.items()):
        ref = _ref(run)
        rows.append({**ref, "run": run, "mH": r["H_frozen_mm"], "mW": r["W_mm"],
                     "mD": r["D_mm"], "mWb": wb.get(run, float("nan")),
                     "mach": r.get("mach_frac", float("nan"))})
    print(f"{'run':6} {'P':>4} {'v':>4} {'SOD':>3} | {'H':>4} {'pred':>5} | {'CL':>5} {'W_fus':>5} | "
          f"{'p':>5} {'D':>5} | {'width':>5} {'W_bead':>6} | mach")
    for r in rows:
        print(f"{r['run']:6} {r['P']:4.1f} {r['v']:4.0f} {r['sod']:3} | {r['H']:4.1f} {r['mH']:5.2f} | "
              f"{r['CL']:5.1f} {r['mW']:5.2f} | {r['p']:5.2f} {r['mD']:5.2f} | {r['Wb']:5.1f} "
              f"{r['mWb']:6.2f} | {r['mach']:.2f}")
    if not rows:
        return

    def mae(pk, rk):
        e = [abs(r[pk] - r[rk]) / r[rk] for r in rows if np.isfinite(r[pk])]
        b = [(r[pk] - r[rk]) / r[rk] for r in rows if np.isfinite(r[pk])]
        return 100 * np.mean(e), 100 * np.mean(b)

    print(f"\nheld-out beads evaluated: {len(rows)}/14")
    for name, pk, rk, lim in (("H vs bead height", "mH", "H", 10), ("W_fus vs contact length", "mW", "CL", 15),
                              ("D vs penetration", "mD", "p", 25), ("W_bead vs width [Amd 1]", "mWb", "Wb", 15)):
        m, b = mae(pk, rk)
        print(f"  {name:26} MAE {m:5.1f} %  (bias {b:+5.1f} %)  limit {lim} %  -> {'PASS' if m <= lim else 'FAIL'}")
    print("\ntrend signs (mean within-group slope; measured vs model)")
    for name, x, rk, pk in (("dH/dv", "v", "H", "mH"), ("d(contact)/dv", "v", "CL", "mW"),
                            ("d(contact)/dP", "P", "CL", "mW"), ("d(width)/dv [Amd 1]", "v", "Wb", "mWb"),
                            ("d(width)/dP [Amd 1]", "P", "Wb", "mWb")):
        sm, n = _slope_sign(rows, x, rk)
        sp, _ = _slope_sign(rows, x, pk)
        ok = np.sign(sm) == np.sign(sp)
        print(f"  {name:22} measured {sm:+.4f}  model {sp:+.4f}  (groups {n})  -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
