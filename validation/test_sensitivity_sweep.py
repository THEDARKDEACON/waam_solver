"""
test_sensitivity_sweep.py — Smoke the calibrate toggle sweep (CI subset).

Runs under ``WAAM_FULL_VALIDATION=1``. Prefer CUDA; uses 4000 steps by default
(indicative ranking — full 8000-step table: ``python3 -m waam_twin.tools.sensitivity_sweep``).
"""

from __future__ import annotations

import os
import sys

from waam_twin.tools.sensitivity_sweep import CI_SUBSET, SWEEP_CASES, run_sweep


def run() -> None:
    os.environ.setdefault("WAAM_BACKEND", "cuda")
    os.environ.setdefault("WAAM_BEAD_STEPS", "4000")
    os.environ.setdefault("WAAM_MAX_BEAD_STEPS", "4000")

    cases = tuple(c for c in SWEEP_CASES if c.name in CI_SUBSET)
    if len(cases) != len(CI_SUBSET):
        raise AssertionError("CI subset cases missing from SWEEP_CASES")

    n_steps = int(os.environ["WAAM_BEAD_STEPS"])
    result = run_sweep(cases=cases, n_steps=n_steps)
    base = result["baseline"]
    by_name = {r["name"]: r for r in result["cases"]}

    if base["n_liquid"] < 10:
        raise AssertionError(
            f"baseline pool did not develop at {n_steps} steps "
            f"(n_liquid={base['n_liquid']})"
        )
    if base["pool_width_mm"] < 0.5:
        raise AssertionError(
            f"baseline pool too narrow at {n_steps} steps: W={base['pool_width_mm']:.2f} mm"
        )

    freeze = by_name["enable_bead_freeze_off"]
    if freeze["sensitivity_pct"] < 1.0 and freeze["n_liquid"] >= base["n_liquid"] * 0.5:
        raise AssertionError(
            "enable_bead_freeze=false should measurably shift W/D or liquid inventory"
        )

    dx = by_name["dx_0p5mm"]
    if abs(dx["dx_mm"] - 0.5) > 0.05:
        raise AssertionError(f"dx case did not run at 0.5 mm (got {dx['dx_mm']})")

    top = result["cases"][0]
    if top["sensitivity_pct"] < 1.0:
        raise AssertionError(
            f"expected at least one toggle with ≥1% W/D shift at {n_steps} steps"
        )

    print(
        f"[sensitivity_sweep] baseline W/D={base['pool_width_mm']:.2f}/"
        f"{base['pool_depth_mm']:.2f} mm  "
        f"top={top['name']} score={top['sensitivity_pct']:.1f}%"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
