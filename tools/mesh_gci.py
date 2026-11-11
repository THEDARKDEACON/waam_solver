"""Grid-convergence analysis for the mesh study (docs/MESH_STUDY_PREREG.md).

Usage: python -m waam_twin.tools.mesh_gci runs/steady/mesh_M1.jsonl [more.jsonl]

Groups records by (job, dx), averages repeats, and applies the Celik et al.
(2008, J. Fluids Eng. 130:078001) GCI procedure on the three finest grids, with
non-uniform refinement ratios. Runs with a speed-cap hit in the measurement
window (mach_frac_window > 0) or non-finite outputs are excluded and listed.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

from waam_twin.paths import PROJECT_ROOT

OUTPUTS = ("W_mm", "D_mm", "H_frozen_mm", "W_bead_mm", "area_ratio")
CONV_PCT = 6.0  # pre-registered: |change 0.4 -> 0.3| <= 6 %, or <= 2 pooled SD


def _celik(phi1: float, phi2: float, phi3: float, r21: float, r32: float) -> dict:
    """phi1 finest. Returns order p, extrapolated value, GCI_fine [%], kind."""
    e21, e32 = phi2 - phi1, phi3 - phi2
    if e21 == 0.0 or e32 == 0.0:
        return {"kind": "flat", "p": float("nan"), "ext": phi1, "gci_pct": 0.0}
    s = math.copysign(1.0, e32 / e21)
    kind = "monotonic" if s > 0 else "oscillatory"
    p = 2.0
    for _ in range(200):
        q = math.log(max(1e-12, (r21 ** p - s) / (r32 ** p - s)))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-8:
            p = p_new
            break
        p = 0.5 * (p + p_new)
    rp = r21 ** p
    ext = (rp * phi1 - phi2) / (rp - 1.0) if rp != 1.0 else float("nan")
    ea = abs((phi1 - phi2) / phi1) if phi1 else float("nan")
    gci = 1.25 * ea / (rp - 1.0) * 100.0 if rp != 1.0 else float("nan")
    if abs(e32) < abs(e21):
        kind = "divergent"  # change grows on refinement
    return {"kind": kind, "p": p, "ext": ext, "gci_pct": gci}


def main(paths: list[str]) -> None:
    groups: dict[tuple, list] = defaultdict(list)
    excluded = []
    for p in paths:
        path = pathlib.Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            job = pathlib.Path(r["job"]).stem.split("_mat")[0]
            dx = round(float(r["dx_mm"]), 3)
            bad = (r.get("mach_frac_window") or 0) > 0 or not all(
                math.isfinite(float(r.get(k, float("nan")))) for k in OUTPUTS[:3])
            if bad:
                excluded.append((job, dx, r.get("mach_frac_window")))
                continue
            groups[(job, dx)].append(r)
    if excluded:
        print("excluded (speed-cap hit in window or NaN):", excluded)
    jobs = sorted({k[0] for k in groups})
    for job in jobs:
        dxs = sorted(dx for (j, dx) in groups if j == job)
        print(f"\n== {job}   grids {dxs}")
        print(f"   {'output':12} " + "  ".join(f"dx{dx:<5} mean±sd" for dx in dxs))
        for out in OUTPUTS:
            means, sds = [], []
            for dx in dxs:
                vals = np.array([float(r.get(out, np.nan)) for r in groups[(job, dx)]])
                means.append(float(np.nanmean(vals)))
                sds.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan"))
            cells = "  ".join(f"{m:6.3f}±{s:5.3f}  " for m, s in zip(means, sds))
            line = f"   {out:12} {cells}"
            if len(dxs) >= 3:
                h1, h2, h3 = dxs[0], dxs[1], dxs[2]
                f1, f2, f3 = means[0], means[1], means[2]
                c = _celik(f1, f2, f3, h2 / h1, h3 / h2)
                pooled = math.sqrt(np.nanmean(np.square(sds[:2]))) if any(
                    math.isfinite(s) for s in sds[:2]) else float("nan")
                chg = (f1 - f2) / f2 * 100.0
                conv = abs(chg) <= CONV_PCT or (math.isfinite(pooled) and abs(f1 - f2) <= 2 * pooled)
                line += (f"| {c['kind']:11} p={c['p']:.2f} GCI={c['gci_pct']:.1f}%  "
                         f"Δ{h2}->{h1}={chg:+.1f}%  sd={pooled:.3f}  "
                         f"{'CONVERGED' if conv else 'not converged'}")
            print(line)
        # width-power slope needs both beads; printed per grid below
    beads = [j for j in jobs]
    if len(beads) == 2:
        lo, hi = beads
        print(f"\nW_bead {hi} − {lo} per grid (width–power response):")
        for dx in sorted({dx for (_, dx) in groups}):
            if (lo, dx) in groups and (hi, dx) in groups:
                a = np.mean([r["W_bead_mm"] for r in groups[(lo, dx)]])
                b = np.mean([r["W_bead_mm"] for r in groups[(hi, dx)]])
                print(f"   dx {dx}: {b - a:+.3f} mm   (measured width +0.5, contact length +1.0)")


if __name__ == "__main__":
    main(sys.argv[1:])
