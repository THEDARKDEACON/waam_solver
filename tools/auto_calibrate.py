"""
auto_calibrate.py — Iterate fitted knobs until pool W/D gate is satisfied.

Only touches *fitted* knobs (η, Goldak axes, evap_cooling_scale, C_acc,
optional marangoni_scale / heat_loss_factor). Predicted closures stay frozen.

Stops early when max(|W−Wref|/Wref, |D−Dref|/Dref) ≤ --gate-pct
(default 25, or WAAM_POOL_GATE_PCT).

Usage (HPC / GPU):
  WAAM_BACKEND=cuda python -m waam_twin.tools.auto_calibrate \\
    --job jobs/examples/bead_physics_accuracy.yaml \\
    --gate-pct 25 --max-trials 24 --write

  # Quick smoke (few steps, loose gate):
  WAAM_BACKEND=cuda python -m waam_twin.tools.auto_calibrate \\
    --job jobs/examples/bead_calibrate.yaml --quick --write

After success, re-check held-outs without retuning:
  python -m waam_twin.tools.prediction_report
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from waam_twin import WAAMTwin
from waam_twin.benchmark import (
    measure_bead_metrics,
    measure_fusion_zone_mm,
    measure_pool_mm,
    pool_error_pct,
)
from waam_twin.job import load_job_config
from waam_twin.physics.arc import goldak_from_job_mm
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel
from waam_twin.validation.gate_thresholds import process_gate_pct
from waam_twin.validation.prediction import extract_locked_physics


@dataclass
class KnobState:
    eta: float
    a_front_mm: float
    a_rear_mm: float
    b_mm: float
    c_mm: float
    evap_cooling_scale: float
    recoil_accommodation: float
    marangoni_scale: float = 1.0
    heat_loss_factor: float = 1.0

    def clone(self) -> "KnobState":
        return copy.deepcopy(self)


@dataclass
class TrialResult:
    trial: int
    knobs: dict[str, float]
    pool_width_mm: float
    pool_depth_mm: float
    bead_height_mm: float
    error_pct: float
    n_steps: int
    wall_s: float
    passed: bool
    pool_metric: str = "fusion_zone"
    pool_width_bbox_mm: float = 0.0
    pool_depth_bbox_mm: float = 0.0
    pool_width_fusion_mm: float = 0.0
    pool_depth_fusion_mm: float = 0.0


@dataclass
class AutoCalibrateReport:
    job: str
    gate_pct: float
    W_ref_mm: float
    D_ref_mm: float
    satisfied: bool
    trials_run: int
    pool_metric: str = "fusion_zone"
    mesh_fingerprint: dict[str, Any] = field(default_factory=dict)
    best: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    calibration_path: str | None = None
    notes: str = ""


def _pool_metric_from_job(job: dict[str, Any]) -> str:
    claim = job.get("claim") or {}
    sim = job.get("simulation") or {}
    return str(claim.get("pool_metric") or sim.get("pool_metric") or "fusion_zone")


def _knobs_from_job(job: dict[str, Any]) -> KnobState:
    proc = job.get("process") or {}
    g = job.get("goldak") or {}
    adv = job.get("advanced_physics") or {}
    cal = {}
    cal_path = job.get("calibration")
    if cal_path:
        from waam_twin.calibration import load_calibration
        c = load_calibration(str(cal_path))
        cal = {
            "eta": c.arc_efficiency if c.arc_efficiency != 1.0 else None,
            "marangoni_scale": c.marangoni_scale,
            "heat_loss_factor": c.heat_loss_factor,
        }
    eta = float(proc.get("arc_efficiency", 0.8))
    if cal.get("eta") is not None:
        eta = float(cal["eta"])
    return KnobState(
        eta=eta,
        a_front_mm=float(g.get("a_front_mm", 2.2)),
        a_rear_mm=float(g.get("a_rear_mm", 4.2)),
        b_mm=float(g.get("b_mm", 3.0)),
        c_mm=float(g.get("c_mm", 1.5)),
        evap_cooling_scale=float(adv.get("evap_cooling_scale", 25.0)),
        recoil_accommodation=float(adv.get("recoil_accommodation", 0.54)),
        marangoni_scale=float(cal.get("marangoni_scale") or 1.0),
        heat_loss_factor=float(cal.get("heat_loss_factor") or 1.0),
    )


def _macro_reference(job: dict[str, Any]) -> tuple[float, float]:
    ref = job.get("reference") or {}
    return (
        float(ref.get("pool_width_mm", 7.0)),
        float(ref.get("pool_depth_mm", 3.0)),
    )


def _apply_knobs(twin: WAAMTwin, job: dict[str, Any], knobs: KnobState) -> None:
    twin.eta = float(knobs.eta)
    twin.marangoni_scale = float(knobs.marangoni_scale)
    twin.evap_cooling_scale = float(knobs.evap_cooling_scale)
    twin.recoil_accommodation = float(knobs.recoil_accommodation)

    h_base = float((job.get("heat_loss") or {}).get("h_conv", twin.h_conv))
    twin.h_conv = h_base * float(knobs.heat_loss_factor)

    goldak_cfg = dict(job.get("goldak") or {})
    goldak_cfg.update({
        "a_front_mm": knobs.a_front_mm,
        "a_rear_mm": knobs.a_rear_mm,
        "b_mm": knobs.b_mm,
        "c_mm": knobs.c_mm,
    })
    if str(job.get("heat_source", "goldak")).lower() in (
        "goldak", "goldak3d", "doubleellipsoid",
    ):
        twin.arc_source = goldak_from_job_mm(twin, goldak_cfg)


def _run_trial(
    job_path: str,
    job: dict[str, Any],
    knobs: KnobState,
    *,
    n_steps: int | None,
    trial: int,
    gate_pct: float,
    W_ref: float,
    D_ref: float,
    backend: str,
) -> TrialResult:
    reset_taichi()
    init_taichi(backend=backend)
    twin = WAAMTwin.from_job(job_path)
    _apply_knobs(twin, job, knobs)
    twin.reset()

    steps, x0, y0, dx_dir = plan_linear_bead_run(twin, job, n_steps=n_steps)
    t0 = time.perf_counter()
    run_bead_travel(twin, steps, x_start_m=x0, y_m=y0, direction_x=dx_dir)
    wall = time.perf_counter() - t0

    W_bbox, D_bbox, _n_liq = measure_pool_mm(twin)
    fusion = measure_fusion_zone_mm(twin)
    W_fus = float(fusion.get("fusion_width_mm", 0.0))
    D_fus = float(fusion.get("fusion_depth_mm", 0.0))
    pool_metric = _pool_metric_from_job(job)
    if pool_metric == "fl_bbox":
        W_mm, D_mm = float(W_bbox), float(D_bbox)
    else:
        W_mm, D_mm = W_fus, D_fus
    metrics = measure_bead_metrics(twin)
    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    return TrialResult(
        trial=trial,
        knobs={
            "eta": knobs.eta,
            "a_front_mm": knobs.a_front_mm,
            "a_rear_mm": knobs.a_rear_mm,
            "b_mm": knobs.b_mm,
            "c_mm": knobs.c_mm,
            "evap_cooling_scale": knobs.evap_cooling_scale,
            "recoil_accommodation": knobs.recoil_accommodation,
            "marangoni_scale": knobs.marangoni_scale,
            "heat_loss_factor": knobs.heat_loss_factor,
        },
        pool_width_mm=float(W_mm),
        pool_depth_mm=float(D_mm),
        bead_height_mm=float(metrics["bead_height_mm"]),
        error_pct=float(err),
        n_steps=int(steps),
        wall_s=float(wall),
        passed=bool(err <= gate_pct),
        pool_metric=pool_metric,
        pool_width_bbox_mm=float(W_bbox),
        pool_depth_bbox_mm=float(D_bbox),
        pool_width_fusion_mm=W_fus,
        pool_depth_fusion_mm=D_fus,
    )


def _write_calibration_yaml(
    best: TrialResult,
    *,
    job_path: str,
    job: dict[str, Any],
    gate_pct: float,
    out_path: Path,
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required to write calibration YAML") from exc

    locked = extract_locked_physics(job)
    m = locked.get("mesh") or {}
    mesh = {
        "mesh_tier": m.get("mesh_tier"),
        "dx_mm": m.get("dx_mm"),
        "dt_scale": m.get("dt_scale"),
        "use_variable_tau": m.get("use_variable_tau"),
        "use_srt": m.get("use_srt"),
        "auto_dt_ma": m.get("auto_dt_ma"),
        "C_darcy": m.get("C_darcy"),
        "u_mach_limit_lu": m.get("u_mach_limit_lu"),
        "force_limit_lu": m.get("force_limit_lu"),
        "u_design_m_s": m.get("u_design_m_s"),
        "pool_metric": m.get("pool_metric") or best.pool_metric,
    }

    data = {
        "material": "materials/validated/ER70S-6.v1.yaml",
        "process": "auto_calibrate",
        "arc_efficiency": round(best.knobs["eta"], 4),
        "heat_loss_factor": round(best.knobs["heat_loss_factor"], 4),
        "marangoni_scale": round(best.knobs["marangoni_scale"], 4),
        "arc_sigma_scale": 1.0,
        "goldak": {
            "a_front_mm": round(best.knobs["a_front_mm"], 3),
            "a_rear_mm": round(best.knobs["a_rear_mm"], 3),
            "b_mm": round(best.knobs["b_mm"], 3),
            "c_mm": round(best.knobs["c_mm"], 3),
        },
        "advanced_physics": {
            "evap_cooling_scale": round(best.knobs["evap_cooling_scale"], 3),
            "recoil_accommodation": round(best.knobs["recoil_accommodation"], 4),
        },
        "mesh_fingerprint": mesh,
        "fit_metrics": {
            "job": job_path,
            "gate_pct": gate_pct,
            "n_steps": best.n_steps,
            "pool_metric": best.pool_metric,
            "pool_width_mm": round(best.pool_width_mm, 3),
            "pool_depth_mm": round(best.pool_depth_mm, 3),
            "pool_width_bbox_mm": round(best.pool_width_bbox_mm, 3),
            "pool_depth_bbox_mm": round(best.pool_depth_bbox_mm, 3),
            "pool_width_fusion_mm": round(best.pool_width_fusion_mm, 3),
            "pool_depth_fusion_mm": round(best.pool_depth_fusion_mm, 3),
            "bead_height_mm": round(best.bead_height_mm, 3),
            "error_pct_macro": round(best.error_pct, 2),
            "passed": best.passed,
        },
        "notes": (
            "Auto-fitted by waam_twin.tools.auto_calibrate. "
            "Copy goldak/advanced_physics/η into the *same mesh tier* calibrate job lock, "
            "then run prediction_report --tier shop|hpc on held-outs without retuning. "
            "Do not reuse knobs across shop/hpc mesh fingerprints."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return out_path


def _patch_job_yaml(job_path: Path, knobs: KnobState) -> None:
    """Write fitted knobs back into an existing job YAML (in place)."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required for --write-job") from exc

    with open(job_path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("process", {})["arc_efficiency"] = round(knobs.eta, 4)
    g = data.setdefault("goldak", {})
    g["a_front_mm"] = round(knobs.a_front_mm, 3)
    g["a_rear_mm"] = round(knobs.a_rear_mm, 3)
    g["b_mm"] = round(knobs.b_mm, 3)
    g["c_mm"] = round(knobs.c_mm, 3)
    adv = data.setdefault("advanced_physics", {})
    adv["evap_cooling_scale"] = round(knobs.evap_cooling_scale, 3)
    adv["recoil_accommodation"] = round(knobs.recoil_accommodation, 4)
    with open(job_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _candidate_schedule(base: KnobState, *, quick: bool) -> list[tuple[str, KnobState]]:
    """Ordered (label, knobs) list — baseline first, then coordinate probes."""
    out: list[tuple[str, KnobState]] = [("baseline", base.clone())]

    def add(label: str, **updates: float) -> None:
        k = base.clone()
        for key, val in updates.items():
            setattr(k, key, float(val))
        out.append((label, k))

    # η: dominant heat lever
    eta_grid = [0.65, 0.70, 0.75, 0.80] if quick else [0.60, 0.68, 0.72, 0.76, 0.82, 0.88]
    for e in eta_grid:
        if abs(e - base.eta) > 1e-6:
            add(f"eta={e:.2f}", eta=e)

    # Goldak width / depth
    b_grid = [2.4, 3.0, 3.6] if quick else [2.2, 2.6, 3.0, 3.4, 3.8]
    for b in b_grid:
        if abs(b - base.b_mm) > 1e-6:
            add(f"b={b:.1f}", b_mm=b)

    c_grid = [1.2, 1.5, 1.8] if quick else [1.0, 1.3, 1.5, 1.8, 2.2]
    for c in c_grid:
        if abs(c - base.c_mm) > 1e-6:
            add(f"c={c:.1f}", c_mm=c)

    # Recoil / evaporative cooling (secondary)
    if not quick:
        for c_acc in (0.15, 0.25, 0.40, 0.54):
            if abs(c_acc - base.recoil_accommodation) > 1e-6:
                add(f"C_acc={c_acc:.2f}", recoil_accommodation=c_acc)
        for ev in (15.0, 25.0, 40.0):
            if abs(ev - base.evap_cooling_scale) > 1e-6:
                add(f"evap={ev:.0f}", evap_cooling_scale=ev)

    return out


def _refine_around(best: KnobState, *, focus: str) -> list[KnobState]:
    """Local ± steps around the current best for one coordinate."""
    ks: list[KnobState] = []
    if focus == "eta":
        for de in (-0.04, -0.02, 0.02, 0.04):
            k = best.clone()
            k.eta = min(0.95, max(0.45, best.eta + de))
            ks.append(k)
    elif focus == "b_mm":
        for db in (-0.3, -0.15, 0.15, 0.3):
            k = best.clone()
            k.b_mm = min(5.0, max(1.5, best.b_mm + db))
            ks.append(k)
    elif focus == "c_mm":
        for dc in (-0.25, -0.1, 0.1, 0.25):
            k = best.clone()
            k.c_mm = min(3.5, max(0.8, best.c_mm + dc))
            ks.append(k)
    return ks


def run_auto_calibrate(
    job_path: str,
    *,
    gate_pct: float | None = None,
    max_trials: int = 24,
    n_steps: int | None = None,
    quick: bool = False,
    backend: str | None = None,
    refine: bool = True,
) -> AutoCalibrateReport:
    gate = float(gate_pct if gate_pct is not None else process_gate_pct(25.0))
    backend = backend or os.environ.get("WAAM_BACKEND", "cuda")
    job = load_job_config(job_path)
    W_ref, D_ref = _macro_reference(job)
    base = _knobs_from_job(job)
    pool_metric = _pool_metric_from_job(job)
    mesh_fp = (extract_locked_physics(job).get("mesh") or {})

    if n_steps is None:
        model = job.get("model_reference") or {}
        default_steps = int(model.get("n_steps", 4000 if quick else 8000))
        n_steps = 2000 if quick else default_steps

    report = AutoCalibrateReport(
        job=job_path,
        gate_pct=gate,
        W_ref_mm=W_ref,
        D_ref_mm=D_ref,
        satisfied=False,
        trials_run=0,
        pool_metric=pool_metric,
        mesh_fingerprint={
            "mesh_tier": mesh_fp.get("mesh_tier"),
            "dx_mm": mesh_fp.get("dx_mm"),
            "dt_scale": mesh_fp.get("dt_scale"),
            "use_variable_tau": mesh_fp.get("use_variable_tau"),
            "use_srt": mesh_fp.get("use_srt"),
            "auto_dt_ma": mesh_fp.get("auto_dt_ma"),
            "pool_metric": mesh_fp.get("pool_metric") or pool_metric,
        },
    )

    schedule = _candidate_schedule(base, quick=quick)
    seen: set[tuple] = set()
    best: TrialResult | None = None
    trial_i = 0

    def _key(k: KnobState) -> tuple:
        return (
            round(k.eta, 4), round(k.b_mm, 3), round(k.c_mm, 3),
            round(k.a_front_mm, 3), round(k.a_rear_mm, 3),
            round(k.evap_cooling_scale, 2), round(k.recoil_accommodation, 3),
            round(k.marangoni_scale, 3), round(k.heat_loss_factor, 3),
        )

    queue: list[tuple[str, KnobState]] = list(schedule)

    print(
        f"[auto_calibrate] job={job_path}\n"
        f"  gate≤{gate:.1f}%  W/D_ref={W_ref:.2f}×{D_ref:.2f} mm  "
        f"pool_metric={pool_metric}  mesh_tier={mesh_fp.get('mesh_tier')}  "
        f"dx_mm={mesh_fp.get('dx_mm')}  "
        f"max_trials={max_trials}  n_steps≈{n_steps}  backend={backend}",
        flush=True,
    )

    while queue and trial_i < max_trials:
        label, knobs = queue.pop(0)
        sig = _key(knobs)
        if sig in seen:
            continue
        seen.add(sig)
        trial_i += 1

        print(f"\n[auto_calibrate] trial {trial_i}/{max_trials}  {label}", flush=True)
        result = _run_trial(
            job_path, job, knobs,
            n_steps=n_steps, trial=trial_i, gate_pct=gate,
            W_ref=W_ref, D_ref=D_ref, backend=backend,
        )
        report.history.append(asdict(result))
        report.trials_run = trial_i

        print(
            f"  → W={result.pool_width_mm:.2f} D={result.pool_depth_mm:.2f} mm  "
            f"(bbox {result.pool_width_bbox_mm:.2f}×{result.pool_depth_bbox_mm:.2f})  "
            f"err={result.error_pct:.1f}%  "
            f"{'PASS' if result.passed else 'fail'}  "
            f"({result.wall_s:.0f}s, {result.n_steps} steps)",
            flush=True,
        )

        if best is None or result.error_pct < best.error_pct:
            best = result
            report.best = asdict(result)
            # Local refine around improving points (still under budget)
            if refine and not result.passed and trial_i < max_trials:
                # Prefer refining the coordinate that most mismatches ref
                w_err = abs(result.pool_width_mm - W_ref) / max(W_ref, 0.1)
                d_err = abs(result.pool_depth_mm - D_ref) / max(D_ref, 0.1)
                focus = "b_mm" if w_err >= d_err else "c_mm"
                # Also nudge η if both large or pool too cold/hot overall
                if result.pool_width_mm + result.pool_depth_mm < 0.5 * (W_ref + D_ref):
                    focus = "eta"
                k_best = KnobState(**{
                    "eta": result.knobs["eta"],
                    "a_front_mm": result.knobs["a_front_mm"],
                    "a_rear_mm": result.knobs["a_rear_mm"],
                    "b_mm": result.knobs["b_mm"],
                    "c_mm": result.knobs["c_mm"],
                    "evap_cooling_scale": result.knobs["evap_cooling_scale"],
                    "recoil_accommodation": result.knobs["recoil_accommodation"],
                    "marangoni_scale": result.knobs["marangoni_scale"],
                    "heat_loss_factor": result.knobs["heat_loss_factor"],
                })
                for k in _refine_around(k_best, focus=focus):
                    queue.insert(0, (f"refine-{focus}", k))
                if focus != "eta":
                    for k in _refine_around(k_best, focus="eta"):
                        queue.append((f"refine-eta", k))

        if result.passed:
            report.satisfied = True
            report.notes = f"Gate {gate:.1f}% met on trial {trial_i} ({label})."
            break

    if best is None:
        report.notes = "No trials completed."
    elif not report.satisfied:
        report.notes = (
            f"Budget exhausted ({trial_i} trials); best err={best.error_pct:.1f}% "
            f"> gate {gate:.1f}%. Consider more trials, finer dx, or looser gate."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--job",
        default="jobs/examples/bead_calibrate.yaml",
        help="Calibration job (default: bead_calibrate.yaml)",
    )
    ap.add_argument(
        "--gate-pct",
        type=float,
        default=None,
        help="Max pool W/D error %% to accept (default: WAAM_POOL_GATE_PCT or 25)",
    )
    ap.add_argument("--max-trials", type=int, default=24, help="Hard cap on bead runs")
    ap.add_argument("--n-steps", type=int, default=None, help="Steps per trial (default from job model_reference)")
    ap.add_argument("--quick", action="store_true", help="Smaller grid + fewer steps")
    ap.add_argument("--no-refine", action="store_true", help="Disable local refine around best")
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write materials/calibration/ER70S-6.auto_calibrate.yaml from best",
    )
    ap.add_argument(
        "--write-job",
        action="store_true",
        help="Also patch fitted knobs into the job YAML in place",
    )
    ap.add_argument(
        "--json",
        default="validation/baselines/auto_calibrate_latest.json",
        help="History / report JSON path",
    )
    ap.add_argument("--backend", default=None, help="cuda|cpu|… (default WAAM_BACKEND)")
    args = ap.parse_args(argv)

    report = run_auto_calibrate(
        args.job,
        gate_pct=args.gate_pct,
        max_trials=args.max_trials,
        n_steps=args.n_steps,
        quick=args.quick,
        backend=args.backend,
        refine=not args.no_refine,
    )

    cal_path = Path("materials/calibration/ER70S-6.auto_calibrate.yaml")
    if args.write and report.best:
        best_tr = TrialResult(**{
            **report.best,
            "knobs": report.best["knobs"],
        })
        # Reconstruct TrialResult carefully
        best_tr = TrialResult(
            trial=int(report.best["trial"]),
            knobs=dict(report.best["knobs"]),
            pool_width_mm=float(report.best["pool_width_mm"]),
            pool_depth_mm=float(report.best["pool_depth_mm"]),
            bead_height_mm=float(report.best["bead_height_mm"]),
            error_pct=float(report.best["error_pct"]),
            n_steps=int(report.best["n_steps"]),
            wall_s=float(report.best["wall_s"]),
            passed=bool(report.best["passed"]),
            pool_metric=str(report.best.get("pool_metric", report.pool_metric)),
            pool_width_bbox_mm=float(report.best.get("pool_width_bbox_mm", 0.0)),
            pool_depth_bbox_mm=float(report.best.get("pool_depth_bbox_mm", 0.0)),
            pool_width_fusion_mm=float(report.best.get("pool_width_fusion_mm", 0.0)),
            pool_depth_fusion_mm=float(report.best.get("pool_depth_fusion_mm", 0.0)),
        )
        job_cfg = load_job_config(args.job)
        path = _write_calibration_yaml(
            best_tr,
            job_path=args.job,
            job=job_cfg,
            gate_pct=report.gate_pct,
            out_path=cal_path,
        )
        report.calibration_path = str(path)
        print(f"[auto_calibrate] wrote {path}", flush=True)

        if args.write_job:
            k = KnobState(**best_tr.knobs)
            job_p = Path(args.job)
            if not job_p.is_file():
                from waam_twin.paths import resolve_project_path
                job_p = Path(resolve_project_path(args.job))
            _patch_job_yaml(job_p, k)
            print(f"[auto_calibrate] patched job knobs → {job_p}", flush=True)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), indent=2))
    print(f"[auto_calibrate] report → {json_path}", flush=True)

    if report.satisfied:
        print(
            f"\n[auto_calibrate] SATISFIED  err={report.best.get('error_pct', 0):.1f}% "
            f"≤ {report.gate_pct:.1f}%  after {report.trials_run} trials",
            flush=True,
        )
        tier = report.mesh_fingerprint.get("mesh_tier") or "shop"
        print(
            "Next: freeze these knobs on held-outs of the *same* mesh tier, then:\n"
            f"  python -m waam_twin.tools.prediction_report --tier {tier}",
            flush=True,
        )
        return 0

    print(
        f"\n[auto_calibrate] NOT SATISFIED  best_err="
        f"{report.best.get('error_pct', float('nan')):.1f}%  "
        f"gate={report.gate_pct:.1f}%  trials={report.trials_run}",
        flush=True,
    )
    print(f"  {report.notes}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
