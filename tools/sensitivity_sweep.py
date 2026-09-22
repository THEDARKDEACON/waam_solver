"""
sensitivity_sweep.py — One-at-a-time job toggle sweep vs macro W/D on the calibrate lock.

Runs ``bead_calibrate.yaml`` as baseline, then replays the same path with a single
YAML knob changed per case. Reports ΔW/ΔD and ranks cases by max relative geometry shift.

Locked knobs (η, Goldak, recoil accommodation, evap scale, …) are **not** swept here —
see ``recoil_accommodation_sweep`` for C_acc evidence.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.sensitivity_sweep
  PYTHONPATH=. WAAM_BEAD_STEPS=4000 python3 -m waam_twin.tools.sensitivity_sweep --quick
  PYTHONPATH=. python3 -m waam_twin.tools.sensitivity_sweep --subset ci --quick
  PYTHONPATH=. python3 -m waam_twin.tools.sensitivity_sweep --json out/sweep.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics, measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel
from waam_twin.validation.prediction import CALIBRATE_JOB


@dataclass(frozen=True)
class SweepCase:
    name: str
    group: str
    path: tuple[str, ...]
    alt: Any
    note: str = ""


# Non-locked knobs that often differ between interactive jobs and the calibrate lock.
SWEEP_CASES: tuple[SweepCase, ...] = (
    SweepCase(
        "enable_bead_freeze_off",
        "simulation",
        ("simulation", "enable_bead_freeze"),
        False,
        "bead_on_plate.yaml leaves this false",
    ),
    SweepCase(
        "trailing_lookback_2p5mm",
        "deposition",
        ("deposition", "trailing_solidify_lookback_mm"),
        2.5,
        "engine default vs calibrate 6 mm",
    ),
    SweepCase(
        "trailing_margin_35K",
        "deposition",
        ("deposition", "trailing_solidify_temp_margin_K"),
        35.0,
        "engine default vs calibrate 10 K",
    ),
    SweepCase(
        "contact_angle_60deg",
        "wetting",
        ("surface_wetting", "contact_angle_deg"),
        60.0,
        "wetter toe",
    ),
    SweepCase(
        "contact_angle_100deg",
        "wetting",
        ("surface_wetting", "contact_angle_deg"),
        100.0,
        "drier toe",
    ),
    SweepCase(
        "deposition_superheat_300K",
        "deposition",
        ("deposition", "superheat_K"),
        300.0,
        "cooler wire entry",
    ),
    SweepCase(
        "h_conv_25",
        "heat_loss",
        ("heat_loss", "h_conv"),
        25.0,
        "lighter convection",
    ),
    SweepCase(
        "h_conv_50",
        "heat_loss",
        ("heat_loss", "h_conv"),
        50.0,
        "stronger convection",
    ),
    SweepCase(
        "radiation_off",
        "heat_loss",
        ("heat_loss", "radiation"),
        False,
        "convection-only losses",
    ),
    SweepCase(
        "arc_surface_weighting_off",
        "simulation",
        ("simulation", "arc_surface_weighting"),
        False,
        "bulk vs surface-biased arc heating",
    ),
    SweepCase(
        "enable_recoil_off",
        "simulation",
        ("simulation", "enable_recoil"),
        False,
        "recoil force disabled (not C_acc)",
    ),
    SweepCase(
        "enable_evap_cooling_off",
        "simulation",
        ("simulation", "enable_evaporative_cooling"),
        False,
        "no Hertz–Knudsen sink",
    ),
    SweepCase(
        "enable_lorentz_off",
        "simulation",
        ("simulation", "enable_lorentz"),
        False,
        "electromagnetic stirring off",
    ),
    SweepCase(
        "enable_gas_shear_off",
        "simulation",
        ("simulation", "enable_gas_shear"),
        False,
        "shielding jet shear off",
    ),
    SweepCase(
        "enable_wetting_off",
        "simulation",
        ("simulation", "enable_wetting"),
        False,
        "contact-angle φ BC off",
    ),
    SweepCase(
        "enable_hydrostatic_off",
        "simulation",
        ("simulation", "enable_hydrostatic_gravity"),
        False,
        "no hydrostatic head",
    ),
    SweepCase(
        "enable_droplet_impact_off",
        "simulation",
        ("simulation", "enable_droplet_impact_pressure"),
        False,
        "droplet momentum only",
    ),
    SweepCase(
        "gas_jet_8ms",
        "advanced",
        ("advanced_physics", "gas_jet_velocity_m_s"),
        8.0,
        "weaker shield gas",
    ),
    SweepCase(
        "gas_jet_16ms",
        "advanced",
        ("advanced_physics", "gas_jet_velocity_m_s"),
        16.0,
        "stronger shield gas",
    ),
    SweepCase(
        "gas_shear_coeff_0p5",
        "advanced",
        ("advanced_physics", "gas_shear_coeff"),
        0.5,
        "half shear coupling",
    ),
    SweepCase(
        "dx_0p5mm",
        "grid",
        ("simulation", "dx_mm"),
        0.5,
        "bead_on_plate coarser dx (0.4 lock)",
    ),
    SweepCase(
        "material_placeholder",
        "material",
        ("material",),
        "materials/placeholders/ER70S-6.yaml",
        "validated v1 vs placeholder file",
    ),
    SweepCase(
        "wire_feed_4p0",
        "process",
        ("process", "wire_feed_m_min"),
        4.0,
        "bead_on_plate WFS vs calibrate 3.2",
    ),
)

CI_SUBSET = (
    "enable_bead_freeze_off",
    "dx_0p5mm",
    "trailing_lookback_2p5mm",
    "material_placeholder",
)


def _get_nested(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _sweep_tmp_path() -> Path:
    tmp_dir = PROJECT_ROOT / ".waam_sweep_tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir / f"waam_sweep_{uuid.uuid4().hex}.yaml"


def _run_job_dict(job: dict[str, Any], n_steps: int) -> dict[str, Any]:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    tmp_path = _sweep_tmp_path()
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(job, fh, sort_keys=False)
        twin = WAAMTwin.from_job(tmp_path)
        twin.reset()
        n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)
        run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)
        W_mm, D_mm, n_liq = measure_pool_mm(twin)
        bead = measure_bead_metrics(twin)
        telem = twin.get_telemetry()
        ref = job.get("reference") or {}
        W_ref = ref.get("pool_width_mm")
        D_ref = ref.get("pool_depth_mm")
        out: dict[str, Any] = {
            "pool_width_mm": float(W_mm),
            "pool_depth_mm": float(D_mm),
            "n_liquid": int(n_liq),
            "bead_height_mm": float(bead.get("bead_height_mm", telem.get("bead_height_mm", 0))),
            "peak_temp_C": float(telem.get("peak_temp_C", 0)),
            "n_steps": int(n_steps),
            "dx_mm": float(twin.grid.dx * 1000.0),
            "grid": f"{twin.grid.nx}×{twin.grid.ny}×{twin.grid.nz}",
        }
        if W_ref is not None and D_ref is not None:
            out["macro_err_pct"] = pool_error_pct(W_mm, D_mm, float(W_ref), float(D_ref))
        return out
    finally:
        tmp_path.unlink(missing_ok=True)


def _sensitivity_score(base: dict[str, Any], alt: dict[str, Any]) -> float:
    Wb = max(base["pool_width_mm"], 0.1)
    Db = max(base["pool_depth_mm"], 0.1)
    dW = abs(alt["pool_width_mm"] - base["pool_width_mm"]) / Wb * 100.0
    dD = abs(alt["pool_depth_mm"] - base["pool_depth_mm"]) / Db * 100.0
    return max(dW, dD)


def run_sweep(
    *,
    cases: tuple[SweepCase, ...],
    n_steps: int,
    baseline_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_job = baseline_job or load_job_config(CALIBRATE_JOB)
    ref = baseline_job.get("reference") or {}
    print(
        f"Sensitivity sweep on {CALIBRATE_JOB}\n"
        f"  macro ref W×D={ref.get('pool_width_mm')}×{ref.get('pool_depth_mm')} mm  "
        f"steps={n_steps}  cases={len(cases)}\n"
    )
    print("Running baseline …")
    base = _run_job_dict(copy.deepcopy(baseline_job), n_steps)
    base_err = base.get("macro_err_pct")
    err_s = f"  macro_err={base_err:.1f}%" if base_err is not None else ""
    print(
        f"  baseline W={base['pool_width_mm']:.2f} D={base['pool_depth_mm']:.2f} mm  "
        f"h={base['bead_height_mm']:.2f}  grid={base['grid']} dx={base['dx_mm']:.3f} mm"
        f"{err_s}\n"
    )
    if base["n_liquid"] < 30:
        print(
            "WARN: baseline pool under-developed (n_liquid < 30). "
            "Use full n_steps≈8000 for ranking decisions."
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        job = copy.deepcopy(baseline_job)
        was = _get_nested(job, case.path)
        _set_nested(job, case.path, case.alt)
        alt = _run_job_dict(job, n_steps)
        score = _sensitivity_score(base, alt)
        dW = alt["pool_width_mm"] - base["pool_width_mm"]
        dD = alt["pool_depth_mm"] - base["pool_depth_mm"]
        row = {
            "name": case.name,
            "group": case.group,
            "path": ".".join(case.path),
            "baseline_value": was,
            "alt_value": case.alt,
            "note": case.note,
            "pool_width_mm": alt["pool_width_mm"],
            "pool_depth_mm": alt["pool_depth_mm"],
            "delta_W_mm": dW,
            "delta_D_mm": dD,
            "sensitivity_pct": score,
            "macro_err_pct": alt.get("macro_err_pct"),
            "bead_height_mm": alt["bead_height_mm"],
            "n_liquid": alt["n_liquid"],
            "grid": alt["grid"],
            "dx_mm": alt["dx_mm"],
        }
        rows.append(row)
        print(f"  {case.name} … score={score:.1f}%")

    rows.sort(key=lambda r: r["sensitivity_pct"], reverse=True)

    print(
        f"\n{'rank':>4}  {'score%':>7}  {'ΔW':>6}  {'ΔD':>6}  "
        f"{'W':>6}  {'D':>6}  {'case':<28}  baseline → alt"
    )
    for i, r in enumerate(rows, 1):
        print(
            f"{i:4d}  {r['sensitivity_pct']:7.1f}  "
            f"{r['delta_W_mm']:+6.2f}  {r['delta_D_mm']:+6.2f}  "
            f"{r['pool_width_mm']:6.2f}  {r['pool_depth_mm']:6.2f}  "
            f"{r['name']:<28}  "
            f"{_format_value(r['baseline_value'])} → {_format_value(r['alt_value'])}"
        )
        if r["note"]:
            print(f"       {r['note']}")

    high = [r for r in rows if r["sensitivity_pct"] >= 15.0]
    low = [r for r in rows if r["sensitivity_pct"] < 3.0]
    print("\n--- interpretation ---")
    print(
        f"Baseline macro err: {base_err:.1f}%" if base_err is not None
        else "Baseline macro err: n/a"
    )
    if high:
        print(
            "High sensitivity (≥15% W or D shift): "
            + ", ".join(r["name"] for r in high)
        )
    if low:
        print(
            "Low sensitivity (<3%): "
            + ", ".join(r["name"] for r in low[:8])
            + (" …" if len(low) > 8 else "")
        )
    print(
        "A wrong high-sensitivity toggle can mimic a bad macro fit; "
        "low-sensitivity toggles are unlikely to explain a ~7% lock error alone."
    )

    return {
        "job": CALIBRATE_JOB,
        "n_steps": n_steps,
        "baseline": base,
        "cases": rows,
        "high_sensitivity": [r["name"] for r in high],
        "low_sensitivity": [r["name"] for r in low],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subset",
        choices=("ci", "all"),
        default="all",
        help="ci = four smoke cases; all = full sweep",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Use WAAM_BEAD_STEPS=4000 if unset (ranking is indicative only)",
    )
    ap.add_argument("--json", type=str, default="", help="Write full results JSON")
    ap.add_argument(
        "--group",
        type=str,
        default="",
        help="Comma-separated groups to include (simulation, deposition, grid, …)",
    )
    args = ap.parse_args(argv)

    if args.quick and "WAAM_BEAD_STEPS" not in os.environ:
        os.environ["WAAM_BEAD_STEPS"] = "4000"
        print("NOTE: --quick uses 4000 steps; prefer full 8000 for lock decisions.")

    job = load_job_config(CALIBRATE_JOB)
    model = job.get("model_reference") or {}
    n_steps = int(os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000)))

    if args.subset == "ci":
        names = set(CI_SUBSET)
        cases = tuple(c for c in SWEEP_CASES if c.name in names)
    else:
        cases = SWEEP_CASES

    if args.group:
        allowed = {g.strip() for g in args.group.split(",") if g.strip()}
        cases = tuple(c for c in cases if c.group in allowed)

    if not cases:
        print("No sweep cases selected.")
        return 1

    t0 = time.perf_counter()
    result = run_sweep(cases=cases, n_steps=n_steps)
    elapsed = time.perf_counter() - t0
    print(f"\nElapsed {elapsed:.0f}s  ({len(cases) + 1} runs)")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {out_path}")

    base = result["baseline"]
    if base["n_liquid"] < 10:
        print("FAIL: baseline did not develop a pool")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
