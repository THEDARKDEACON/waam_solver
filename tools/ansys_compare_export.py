"""
ansys_compare_export.py — Run a bead job and write Ansys-comparable metrics JSON.

Produces the same macro/micro envelope as
``docs/validation/ansys_2024r2/comparison/metrics_schema.json`` so Tier A
Mechanical results can be diffed with
``docs/validation/ansys_2024r2/scripts/compare_to_twin.py``.

Usage:
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.ansys_compare_export
  PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.ansys_compare_export \\
    --job jobs/examples/bead_on_plate.yaml \\
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
from waam_twin.benchmark import measure_bead_metrics, measure_pool_mm
from waam_twin.job import load_job_config
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel

DEFAULT_JOB = "jobs/examples/bead_on_plate.yaml"
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
        model = job.get("model_reference") or {}
        if n_steps is None:
            n_steps = int(
                os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000))
            )
        n_steps, x_start, y_m, dir_x = plan_linear_bead_run(
            twin, job, n_steps=n_steps
        )
        run_bead_travel(
            twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x
        )
        W_mm, D_mm, n_liq = measure_pool_mm(twin)
        bead = measure_bead_metrics(twin)
        telem = twin.get_telemetry()
        proc = job.get("process") or {}
        ref = job.get("reference") or {}
        return {
            "code": "waam_twin",
            "tier": str((job.get("simulation") or {}).get("physics_tier", "full")),
            "source_job": job_path,
            "n_steps": int(n_steps),
            "dx_mm": float(twin.grid.dx * 1000.0),
            "grid": f"{twin.grid.nx}×{twin.grid.ny}×{twin.grid.nz}",
            "material": job.get("material"),
            "material_override": bool(validated_material),
            "process": {
                "current_A": proc.get("current_A"),
                "voltage_V": proc.get("voltage_V"),
                "arc_efficiency": proc.get("arc_efficiency"),
                "travel_speed_mm_s": proc.get("travel_speed_mm_s"),
            },
            "macro": {
                "pool_width_mm": float(W_mm),
                "pool_depth_mm": float(D_mm),
                "bead_height_mm": float(
                    bead.get("bead_height_mm", telem.get("bead_height_mm", 0))
                ),
                "n_liquid": int(n_liq),
                "peak_temp_C": float(telem.get("peak_temp_C", 0)),
                "definition": "measure_pool_mm liquid/remelt envelope",
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
        f"  twin W={m['pool_width_mm']:.2f} D={m['pool_depth_mm']:.2f} mm  "
        f"h={m.get('bead_height_mm', 0):.2f}  peakT={m.get('peak_temp_C', 0):.0f} C"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
