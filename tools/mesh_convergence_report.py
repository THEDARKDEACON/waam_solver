"""
mesh_convergence_report.py — Frozen-knob multi-dx mesh study (Level A toward B).

Runs the same calibrate process / fitted knobs at several ``dx_mm`` values and
reports ΔW/D, bead height, and whether force/Mach clamps fired. Knobs are
**not** retuned — that is the point.

Equal physical travel distance is used across meshes (fair when ``auto_dt_ma``
changes ``dt``).

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.mesh_convergence_report \\
    --job jobs/examples/bead_calibrate_hpc.yaml

  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.mesh_convergence_report \\
    --job jobs/examples/bead_calibrate_hpc.yaml --dx 0.25,0.20,0.15 --gate-pct 10

  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.mesh_convergence_report \\
    --quick --json validation/baselines/mesh_convergence_latest.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics, measure_fusion_zone_mm, measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel, steps_for_travel
from waam_twin.validation.prediction import CALIBRATE_HPC_JOB


# Knobs that must stay identical across the dx series (logged for audit).
_FROZEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("process", "arc_efficiency"),
    ("process", "current_A"),
    ("process", "voltage_V"),
    ("process", "travel_speed_mm_s"),
    ("process", "wire_feed_m_min"),
    ("goldak", "a_front_mm"),
    ("goldak", "a_rear_mm"),
    ("goldak", "b_mm"),
    ("goldak", "c_mm"),
    ("goldak", "ff"),
    ("goldak", "fr"),
    ("advanced_physics", "recoil_accommodation"),
    ("advanced_physics", "evap_cooling_scale"),
    ("simulation", "force_limit_lu"),
    ("simulation", "u_mach_limit_lu"),
    ("simulation", "use_srt"),
    ("simulation", "use_variable_tau"),
    ("simulation", "auto_dt_ma"),
    ("simulation", "C_darcy"),
    ("simulation", "physics_tier"),
)


def _get_nested(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _frozen_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in _FROZEN_PATHS:
        out[".".join(path)] = _get_nested(job, path)
    return out


def _tmp_job_path() -> Path:
    tmp_dir = PROJECT_ROOT / ".waam_sweep_tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir / f"waam_meshconv_{uuid.uuid4().hex}.yaml"


def _run_dx(
    base_job: dict[str, Any],
    dx_mm: float,
    *,
    travel_distance_m: float | None,
    n_steps_override: int | None,
    backend: str | None,
) -> dict[str, Any]:
    job = copy.deepcopy(base_job)
    sim = job.setdefault("simulation", {})
    sim["dx_mm"] = float(dx_mm)
    # Mesh study is not a held-out lock check — drop claim absolute flag noise.
    claim = job.get("claim")
    if isinstance(claim, dict):
        claim = dict(claim)
        claim["notes"] = (
            str(claim.get("notes") or "")
            + f" | mesh_convergence dx_mm={dx_mm:g} (frozen knobs)"
        ).strip(" |")
        job["claim"] = claim

    reset_taichi()
    init_taichi(backend=backend or os.environ.get("WAAM_BACKEND", "cuda"))
    tmp_path = _tmp_job_path()
    t0 = time.perf_counter()
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(job, fh, sort_keys=False)
        twin = WAAMTwin.from_job(tmp_path)
        twin.reset()

        if travel_distance_m is not None and travel_distance_m > 0.0:
            n_steps = steps_for_travel(
                travel_distance_m, twin.travel_speed_m_s, twin.grid.dt
            )
            max_steps = os.environ.get("WAAM_MAX_BEAD_STEPS")
            if max_steps:
                n_steps = min(n_steps, int(max_steps))
            if n_steps_override is not None:
                n_steps = min(n_steps, int(n_steps_override))
            x_start = max(0.004, 4 * twin.grid.dx)
            y_m = (twin.grid.ny // 2) * twin.grid.dx
            dir_x = 1.0
            # Prefer path-planned start if available.
            planned = plan_linear_bead_run(twin, job, n_steps=n_steps)
            x_start, y_m, dir_x = planned[1], planned[2], planned[3]
        else:
            model = job.get("model_reference") or {}
            default_steps = int(
                n_steps_override
                or os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000))
            )
            n_steps, x_start, y_m, dir_x = plan_linear_bead_run(
                twin, job, n_steps=default_steps
            )

        run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

        W_bbox, D_bbox, n_liq = measure_pool_mm(twin)
        fusion = measure_fusion_zone_mm(twin)
        W_fus = float(fusion.get("fusion_width_mm", 0.0))
        D_fus = float(fusion.get("fusion_depth_mm", 0.0))
        claim = job.get("claim") or {}
        pool_metric = str(
            claim.get("pool_metric")
            or sim.get("pool_metric")
            or "fusion_zone"
        )
        if pool_metric == "fl_bbox":
            W_mm, D_mm = W_bbox, D_bbox
        else:
            W_mm, D_mm = W_fus, D_fus
        bead = measure_bead_metrics(twin)
        telem = twin.get_telemetry()
        ref = job.get("reference") or {}

        force_steps = int(telem.get("force_clamp_steps", 0) or 0)
        mach_steps = int(telem.get("mach_clamp_steps", 0) or 0)
        out: dict[str, Any] = {
            "dx_mm": float(twin.grid.dx * 1000.0),
            "requested_dx_mm": float(dx_mm),
            "grid": f"{twin.grid.nx}×{twin.grid.ny}×{twin.grid.nz}",
            "n_cells": int(twin.grid.nx * twin.grid.ny * twin.grid.nz),
            "dt_s": float(twin.grid.dt),
            "dt_scale": float(getattr(twin.grid, "dt_scale", 1.0) or 1.0),
            "auto_dt_ma": bool(getattr(twin, "auto_dt_ma", False)),
            "n_steps": int(n_steps),
            "travel_distance_mm": float(
                n_steps * twin.travel_speed_m_s * twin.grid.dt * 1000.0
            ),
            "pool_metric": pool_metric,
            "pool_width_mm": float(W_mm),
            "pool_depth_mm": float(D_mm),
            "pool_width_bbox_mm": float(W_bbox),
            "pool_depth_bbox_mm": float(D_bbox),
            "pool_width_fusion_mm": W_fus,
            "pool_depth_fusion_mm": D_fus,
            "n_liquid": int(n_liq),
            "bead_height_mm": float(
                bead.get("bead_height_mm", telem.get("bead_height_mm", 0))
            ),
            "peak_temp_C": float(telem.get("peak_temp_C", 0)),
            "eta": float(twin.eta),
            "recoil_accommodation": float(twin.recoil_accommodation),
            "force_limit_lu": float(telem.get("force_limit_lu", 0)),
            "u_mach_limit_lu": float(telem.get("u_mach_limit_lu", 0)),
            "force_clamp_steps": force_steps,
            "mach_clamp_steps": mach_steps,
            "force_clamp_hits_cum": int(telem.get("force_clamp_hits_cum", 0) or 0),
            "mach_clamp_hits_cum": int(telem.get("mach_clamp_hits_cum", 0) or 0),
            "clamps_active": bool(force_steps > 0 or mach_steps > 0),
            "wall_s": float(time.perf_counter() - t0),
        }
        W_ref, D_ref = ref.get("pool_width_mm"), ref.get("pool_depth_mm")
        if W_ref is not None and D_ref is not None and not ref.get("awaiting_measurement"):
            out["macro_err_pct"] = pool_error_pct(
                W_mm, D_mm, float(W_ref), float(D_ref)
            )
            out["macro_W_ref"] = float(W_ref)
            out["macro_D_ref"] = float(D_ref)
        return out
    finally:
        tmp_path.unlink(missing_ok=True)


def _span_pct(values: list[float]) -> float:
    vals = [float(v) for v in values if v == v]  # drop NaN
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if abs(mean) < 1e-9:
        return 0.0
    return (max(vals) - min(vals)) / abs(mean) * 100.0


def _verdict(rows: list[dict[str, Any]], gate_pct: float) -> dict[str, Any]:
    if not rows:
        return {
            "status": "INCONCLUSIVE",
            "reason": "no runs",
            "band_ok": False,
            "clamps_ok": False,
        }
    under = [r for r in rows if int(r.get("n_liquid", 0)) < 30]
    clamps = [r for r in rows if r.get("clamps_active")]
    span_W = _span_pct([float(r["pool_width_mm"]) for r in rows])
    span_D = _span_pct([float(r["pool_depth_mm"]) for r in rows])
    span_h = _span_pct([float(r["bead_height_mm"]) for r in rows])
    span_max = max(span_W, span_D)
    band_ok = span_max <= float(gate_pct)
    clamps_ok = len(clamps) == 0

    if under:
        status = "INCONCLUSIVE"
        reason = (
            f"under-developed pool on dx={[r['dx_mm'] for r in under]} "
            "(n_liquid < 30); raise steps / drop --quick"
        )
    elif not clamps_ok:
        status = "FAIL_CLAMPS"
        reason = (
            "force/Mach clamps active on "
            f"dx={[r['dx_mm'] for r in clamps]} — not a clean convergence study"
        )
    elif not band_ok:
        status = "FAIL_BAND"
        reason = (
            f"max(span_W, span_D)={span_max:.1f}% > gate {gate_pct:g}% "
            "(frozen knobs; Level A not met)"
        )
    else:
        status = "PASS_BAND"
        reason = (
            f"max(span_W, span_D)={span_max:.1f}% ≤ gate {gate_pct:g}% "
            "and clamps inactive (practical mesh band)"
        )

    return {
        "status": status,
        "reason": reason,
        "band_ok": band_ok,
        "clamps_ok": clamps_ok,
        "span_W_pct": span_W,
        "span_D_pct": span_D,
        "span_h_pct": span_h,
        "span_max_pct": span_max,
        "gate_pct": float(gate_pct),
    }


def run_mesh_convergence(
    job_path: str,
    dx_list_mm: list[float],
    *,
    gate_pct: float = 10.0,
    n_steps: int | None = None,
    quick: bool = False,
    backend: str | None = None,
) -> dict[str, Any]:
    base = load_job_config(job_path)
    frozen = _frozen_snapshot(base)
    ref = base.get("reference") or {}
    model = base.get("model_reference") or {}

    if quick and n_steps is None:
        n_steps = int(os.environ.get("WAAM_BEAD_STEPS", "4000"))
        if "WAAM_BEAD_STEPS" not in os.environ:
            os.environ["WAAM_BEAD_STEPS"] = str(n_steps)

    # Establish travel distance from the first (usually coarsest) mesh so finer
    # dx / auto_dt_ma still cover the same physical bead length.
    dx_sorted = sorted(float(x) for x in dx_list_mm)
    print(
        f"[mesh_convergence] job={job_path}\n"
        f"  dx_mm={dx_sorted}  gate_pct={gate_pct:g}  frozen knobs (no retune)\n"
        f"  ref W×D={ref.get('pool_width_mm')}×{ref.get('pool_depth_mm')} mm  "
        f"model_n_steps={model.get('n_steps')}\n",
        flush=True,
    )
    print("Frozen knobs:", flush=True)
    for k, v in frozen.items():
        print(f"  {k}: {v}", flush=True)

    rows: list[dict[str, Any]] = []
    travel_m: float | None = None

    for i, dx in enumerate(dx_sorted):
        print(f"\n=== dx_mm={dx:g} ({i + 1}/{len(dx_sorted)}) ===", flush=True)
        row = _run_dx(
            base,
            dx,
            travel_distance_m=travel_m,
            n_steps_override=n_steps,
            backend=backend,
        )
        if travel_m is None:
            travel_m = float(row["travel_distance_mm"]) / 1000.0
        rows.append(row)
        clamp_s = (
            f"  clamps: force_steps={row['force_clamp_steps']} "
            f"mach_steps={row['mach_clamp_steps']}"
        )
        err = row.get("macro_err_pct")
        err_s = f"  macro_err={err:.1f}%" if err is not None else ""
        print(
            f"  grid={row['grid']}  dt={row['dt_s']:.3e}s  steps={row['n_steps']}  "
            f"dist={row['travel_distance_mm']:.2f}mm\n"
            f"  W={row['pool_width_mm']:.2f}  D={row['pool_depth_mm']:.2f}  "
            f"h={row['bead_height_mm']:.2f}  n_liq={row['n_liquid']}  "
            f"T={row['peak_temp_C']:.0f}C{err_s}\n"
            f"{clamp_s}  wall={row['wall_s']:.1f}s",
            flush=True,
        )

    verdict = _verdict(rows, gate_pct)
    report: dict[str, Any] = {
        "tool": "mesh_convergence_report",
        "level": "A_practical_band",
        "toward": "B_twin_continuum_limit",
        "job": job_path,
        "dx_mm": dx_sorted,
        "frozen_knobs": frozen,
        "equal_travel_distance_mm": (
            None if travel_m is None else float(travel_m * 1000.0)
        ),
        "runs": rows,
        "verdict": verdict,
        "interpretation": (
            "PASS_BAND = practical independence in this dx band (Level A). "
            "FAIL_BAND = mesh-locked knobs still absorb grid error — keep locks, "
            "invest in interface/LBM scaling before claiming B. "
            "FAIL_CLAMPS = raise force/Ma limits or refine dt scaling until clamps "
            "are inactive, then re-run. "
            "C (match experiment at all meshes) requires B first."
        ),
    }

    print("\n--- mesh convergence ---", flush=True)
    print(
        f"{'dx_mm':>7}  {'W':>6}  {'D':>6}  {'h':>6}  {'n_liq':>6}  "
        f"{'Fclamp':>6}  {'Mclamp':>6}  {'macro%':>7}",
        flush=True,
    )
    for r in rows:
        err = r.get("macro_err_pct")
        err_s = f"{err:7.1f}" if err is not None else f"{'n/a':>7}"
        print(
            f"{r['dx_mm']:7.3f}  {r['pool_width_mm']:6.2f}  {r['pool_depth_mm']:6.2f}  "
            f"{r['bead_height_mm']:6.2f}  {r['n_liquid']:6d}  "
            f"{r['force_clamp_steps']:6d}  {r['mach_clamp_steps']:6d}  {err_s}",
            flush=True,
        )
    v = verdict
    print(
        f"\nspan_W={v['span_W_pct']:.1f}%  span_D={v['span_D_pct']:.1f}%  "
        f"span_h={v['span_h_pct']:.1f}%  gate={gate_pct:g}%\n"
        f"VERDICT: {v['status']} — {v['reason']}",
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--job",
        default=CALIBRATE_HPC_JOB,
        help="Baseline job (knobs frozen from this YAML)",
    )
    ap.add_argument(
        "--dx",
        default="0.30,0.25,0.20",
        help="Comma-separated dx_mm list (default: 0.30,0.25,0.20)",
    )
    ap.add_argument(
        "--gate-pct",
        type=float,
        default=10.0,
        help="Max allowed (max-min)/mean %% on W or D across the dx series",
    )
    ap.add_argument("--n-steps", type=int, default=None, help="Cap steps per mesh")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode (~4000 steps); may yield INCONCLUSIVE",
    )
    ap.add_argument(
        "--json",
        default="validation/baselines/mesh_convergence_latest.json",
        help="Write report JSON",
    )
    ap.add_argument("--backend", default=None, help="cuda|cpu|…")
    args = ap.parse_args(argv)

    dx_list = [float(x.strip()) for x in str(args.dx).split(",") if x.strip()]
    if len(dx_list) < 2:
        print("Need at least two --dx values", file=sys.stderr)
        return 2

    report = run_mesh_convergence(
        args.job,
        dx_list,
        gate_pct=args.gate_pct,
        n_steps=args.n_steps,
        quick=args.quick,
        backend=args.backend,
    )

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[mesh_convergence] report → {json_path}", flush=True)

    status = report["verdict"]["status"]
    if status == "PASS_BAND":
        return 0
    if status == "INCONCLUSIVE":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
