"""
test_mesh_study_gci.py — GCI / observed-order maths in tools/mesh_study.

Synthetic φ(h) = φ0 + C h^p on the non-uniform series used by the study
(0.4, 0.3, 0.2 mm) must return p, φ0 to 1e-6; an oscillatory series must be
flagged instead of given an order; the report must pass/fail the gate from
the error at the target dx.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from waam_twin.tools.mesh_study import observed_order, report


def run() -> float:
    h = (0.2, 0.3, 0.4)
    worst = 0.0
    for p_true in (1.0, 1.5, 2.0):
        phi = tuple(5.0 + 3.0 * hh ** p_true for hh in h)
        res = observed_order(h, phi)
        err = max(abs(res["p"] - p_true), abs(res["phi_ext"] - 5.0))
        worst = max(worst, err)
        print(f"[mesh_gci] p_true={p_true}  p={res['p']:.6f}  φ_ext={res['phi_ext']:.6f}  "
              f"GCI={res['gci_fine_pct']:.2f}%")
        if res["convergence"] != "monotone" or err > 1e-6:
            raise AssertionError(f"observed order wrong for p={p_true}: {res}")
    osc = observed_order(h, (5.0, 5.2, 4.9))
    if osc["convergence"] != "oscillatory" or not math.isnan(osc["p"]):
        raise AssertionError(f"oscillatory series not flagged: {osc}")

    # Report gate on a synthetic results file: W converges (p=2), D has a
    # large 0.4 mm error.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "study.jsonl"
        with out.open("w") as fh:
            for dx in (0.4, 0.3, 0.2):
                fh.write(json.dumps({
                    "dx_mm": dx, "W_mm": 6.0 + 2.0 * dx ** 2, "D_mm": 1.8 + 3.0 * dx ** 1,
                    "H_frozen_mm": 2.0, "area_mm2": 9.0, "steady": True,
                    "nu_ratio": 0.5, "mach_frac": 0.0, "force_clamp_frac": 0.0,
                    "tau": 0.5007, "n_cells": 1000,
                }) + "\n")
        summary = report(out, target_dx=0.4, gate_pct=10.0)
    w_err = summary["W_mm"]["err_at_target_pct"]
    d_err = summary["D_mm"]["err_at_target_pct"]
    if not (abs(w_err - 100 * 0.32 / 6.0) < 1e-3 and d_err > 10.0 and summary["gate_pass"] is False):
        raise AssertionError(f"report gate wrong: W {w_err} D {d_err} pass={summary['gate_pass']}")
    if "TAU" not in summary["flags"].get(0.4, []):
        raise AssertionError(f"τ_eff 0.50105 should be flagged TAU: {summary['flags']}")
    print("[mesh_gci] PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
