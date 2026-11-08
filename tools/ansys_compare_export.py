"""
ansys_compare_export.py — Run a bead job and write Ansys-comparable metrics JSON.

Tier A default: conduction-only Park job + T_max liquidus fusion envelope
(same idea as Ansys Mechanical Max Over Time T >= T_liq). Diff with
``docs/validation/ansys_2024r2/scripts/compare_to_twin.py``.

Uses the job torch path (``TorchPathDriver``) so +Y Park/Ansys lines are
honoured, and derives step count from path length / travel / dt (no silent
8000-step over-heat). Callers can still pass ``--steps`` or ``bead_calibrate.yaml``.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.ansys_compare_export
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.ansys_compare_export \\
    --job jobs/examples/bead_calibrate.yaml \\
    --out docs/validation/ansys_2024r2/comparison/twin_metrics.json \\
    --validated-material
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path

import yaml

from waam_twin import WAAMTwin
from waam_twin.benchmark import (
    measure_bead_metrics,
    measure_fusion_zone_mm,
    measure_pool_mm,
)
from waam_twin.job import load_job_config, parse_torch_path
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.torch_path import TorchPathDriver
from waam_twin.validation.bead_helpers import (
    plan_linear_bead_run,
    plan_path_run_steps,
    run_bead_travel,
)

DEFAULT_JOB = "jobs/examples/bead_calibrate_thermal.yaml"
VALIDATED_MAT = "materials/validated/ER70S-6.v1.yaml"
DEFAULT_OUT = "docs/validation/ansys_2024r2/comparison/twin_metrics.json"


def _job_path_for_run(job_path: str, validated_material: bool) -> tuple[str, Path | None]:
    """Return (path_for_from_job, optional_temp_to_delete)."""
    if not validated_material:
        return job_path, None
    job = copy.deepcopy(load_job_config(job_path))
    job["material"] = VALIDATED_MAT
    tmp_dir = PROJECT_ROOT / ".waam_sweep_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp = tmp_dir / f"ansys_compare_{uuid.uuid4().hex}.yaml"
    tmp.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    try:
        rel = str(tmp.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(tmp)
    return rel, tmp


def _resolve_n_steps(job: dict, n_steps: int | None) -> int | None:
    """Explicit --steps / WAAM_BEAD_STEPS / model_reference.n_steps; else None (derive)."""
    if n_steps is not None:
        return int(n_steps)
    env = os.environ.get("WAAM_BEAD_STEPS")
    if env:
        return int(env)
    model = job.get("model_reference") or {}
    if model.get("n_steps") is not None:
        return int(model["n_steps"])
    return None


def _run_torch_path(twin, job: dict, n_steps: int | None) -> int:
    """Drive the job CSV/YAML path for the path duration (Ansys-aligned +Y bead)."""
    waypoints = parse_torch_path(job)
    n_steps = plan_path_run_steps(twin, job, n_steps=n_steps)
    driver = TorchPathDriver(waypoints, twin.travel_speed_m_s)
    use_z = bool(getattr(twin, "use_torch_z", False))
    for _step, x, y, z in driver.positions_for_steps(n_steps, twin.grid.dt):
        twin.step(x, y, is_welding=True, torch_z_m=z if use_z else None)
    return n_steps


def run_export(
    job_path: str,
    *,
    n_steps: int | None,
    validated_material: bool,
) -> dict:
    reset_taichi()
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    run_path, tmp = _job_path_for_run(job_path, validated_material)
    try:
        job = load_job_config(run_path)
        twin = WAAMTwin.from_job(run_path)
        twin.reset()
        n_steps = _resolve_n_steps(job, n_steps)
        waypoints = parse_torch_path(job)
        if len(waypoints) >= 2:
            n_steps = _run_torch_path(twin, job, n_steps)
        else:
            n_steps, x_start, y_m, dir_x = plan_linear_bead_run(
                twin, job, n_steps=n_steps
            )
            run_bead_travel(
                twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x
            )
        fusion = measure_fusion_zone_mm(twin)
        W_mm = float(fusion["fusion_width_mm"])
        D_mm = float(fusion["fusion_depth_mm"])
        W_live, D_live, n_liq = measure_pool_mm(twin)
        bead = measure_bead_metrics(twin)
        telem = twin.get_telemetry()
        proc = job.get("process") or {}
        ref = job.get("reference") or {}
        T_peak_K = float(twin.grid.T_max.to_numpy().max())
        return {
            "code": "waam_twin",
            "tier": str((job.get("simulation") or {}).get("physics_tier", "full")),
            "source_job": job_path,
            "n_steps": int(n_steps),
            "dx_mm": float(twin.grid.dx * 1000.0),
            "grid": f"{twin.grid.nx}\u00d7{twin.grid.ny}\u00d7{twin.grid.nz}",
            "material": job.get("material"),
            "material_override": bool(validated_material),
            "process": {
                "current_A": proc.get("current_A"),
                "voltage_V": proc.get("voltage_V"),
                "arc_efficiency": proc.get("arc_efficiency"),
                "travel_speed_mm_s": proc.get("travel_speed_mm_s"),
            },
            "macro": {
                "pool_width_mm": W_mm,
                "pool_depth_mm": D_mm,
                "bead_height_mm": float(
                    bead.get("bead_height_mm", telem.get("bead_height_mm", 0))
                ),
                "n_fusion_cells": int(fusion.get("n_fusion_cells", 0)),
                "T_threshold_K": float(fusion.get("T_threshold_K", 0.0)),
                "peak_temp_C": float(T_peak_K - 273.15),
                "definition": "T_max >= T_liquidus fusion envelope (Ansys Tier A comparable)",
                "pool_width_live_mm": float(W_live),
                "pool_depth_live_mm": float(D_live),
                "n_liquid": int(n_liq),
            },
            "micro": {
                "probe_peak_temp_C": None,
                "t85_s": None,
                "time_above_800C_s": None,
                "time_above_1100C_s": None,
                "note": "Fill from probe recorder / VTK HAZ if needed",
            },
            "experiment": {
                "pool_width_mm": ref.get("pool_width_mm"),
                "pool_depth_mm": ref.get("pool_depth_mm"),
                "source": ref.get("source"),
            },
        }
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", default=DEFAULT_JOB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--validated-material",
        action="store_true",
        help=f"Force {VALIDATED_MAT} instead of job placeholder",
    )
    args = ap.parse_args(argv)

    result = run_export(
        args.job,
        n_steps=args.steps,
        validated_material=args.validated_material,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    m = result["macro"]
    print(f"Wrote {out}")
    print(
        f"  twin fusion W={m['pool_width_mm']:.2f} D={m['pool_depth_mm']:.2f} mm  "
        f"live W={m.get('pool_width_live_mm', 0):.2f} D={m.get('pool_depth_live_mm', 0):.2f}  "
        f"peakT={m.get('peak_temp_C', 0):.0f} C  steps={result['n_steps']}  "
        f"def={m.get('definition', '')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
