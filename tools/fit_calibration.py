"""
fit_calibration.py — Grid-search calibration vs macrograph with full physics stack.

Usage:
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.fit_calibration
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.fit_calibration --write
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.fit_calibration --quick --write
"""

from __future__ import annotations

import argparse
import json
import pathlib

from waam_twin import WAAMTwin
from waam_twin.benchmark import bead_error_pct, measure_bead_metrics
from waam_twin.job import apply_job_to_twin, load_job_config
from waam_twin.platform import init_taichi
from waam_twin.validation.bead_helpers import run_bead_travel

_JOB = "jobs/examples/bead_on_plate.yaml"
_CAL_OUT = "materials/calibration/ER70S-6.bead_on_plate.yaml"


def _macro_reference(job: dict) -> tuple[float, float]:
    ref = job.get("reference", {})
    return (
        float(ref.get("pool_width_mm", 7.0)),
        float(ref.get("pool_depth_mm", 2.8)),
    )


def run_benchmark(
    eta: float,
    sigma_scale: float,
    marangoni_scale: float = 1.0,
    heat_loss_factor: float = 1.0,
    n_steps: int = 2800,
) -> tuple[dict[str, float], float]:
    """Run bead-on-plate with job physics flags (VOF, deposition, freeze, Goldak)."""
    job = load_job_config(_JOB)
    W_ref, D_ref = _macro_reference(job)
    proc = job.get("process", {})
    h_conv_base = float(job.get("heat_loss", {}).get("h_conv", 35.0))

    twin = WAAMTwin.from_job(_JOB, preset_override="standard")
    apply_job_to_twin(twin, job)
    twin.eta = eta
    twin.marangoni_scale = marangoni_scale
    twin.sigma_cells = (0.002 / twin.grid.dx) * sigma_scale
    twin.h_conv = h_conv_base * heat_loss_factor
    twin.reset()

    run_bead_travel(twin, n_steps)
    metrics = measure_bead_metrics(twin)
    if metrics["n_liquid_cells"] < 3 and metrics["n_deposited_cells"] < 10:
        return metrics, 999.0
    err = bead_error_pct(metrics, W_ref, D_ref)
    return metrics, err


def grid_search(
    eta_vals: list[float] | None = None,
    sigma_vals: list[float] | None = None,
    marangoni_vals: list[float] | None = None,
    heat_vals: list[float] | None = None,
    n_steps: int = 2800,
) -> dict:
    eta_vals = eta_vals or [0.72, 0.80, 0.88]
    sigma_vals = sigma_vals or [0.8, 1.0, 1.2]
    marangoni_vals = marangoni_vals or [0.8, 1.0, 1.2]
    heat_vals = heat_vals or [0.8, 1.0, 1.2]
    best = {"error_pct": 1e9}

    for eta in eta_vals:
        for sig in sigma_vals:
            for msc in marangoni_vals:
                for hlf in heat_vals:
                    m, err = run_benchmark(eta, sig, msc, hlf, n_steps=n_steps)
                    print(
                        f"  η={eta:.2f} σ×={sig:.2f} Ma×={msc:.2f} h×={hlf:.2f}  "
                        f"W={m['pool_width_mm']:.2f} D={m['pool_depth_mm']:.2f}  "
                        f"bead_h={m['bead_height_mm']:.2f}  err={err:.1f}%",
                        flush=True,
                    )
                    if err < best["error_pct"]:
                        best = {
                            "error_pct": err,
                            "eta": eta,
                            "sigma_scale": sig,
                            "marangoni_scale": msc,
                            "heat_loss_factor": hlf,
                            "W_mm": m["pool_width_mm"],
                            "D_mm": m["pool_depth_mm"],
                            "bead_h_mm": m["bead_height_mm"],
                            "bead_w_mm": m["bead_width_mm"],
                        }
    return best


def write_calibration_yaml(best: dict, path: str = _CAL_OUT) -> pathlib.Path:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required") from exc

    out = pathlib.Path(path)
    data = {
        "material": "materials/validated/ER70S-6.v1.yaml",
        "process": "bead_on_plate_macrograph",
        "arc_efficiency": round(best["eta"], 4),
        "heat_loss_factor": round(best["heat_loss_factor"], 4),
        "marangoni_scale": round(best["marangoni_scale"], 4),
        "arc_sigma_scale": round(best["sigma_scale"], 4),
        "fit_metrics": {
            "preset": "standard",
            "physics": "full_vof_deposition_freeze_goldak",
            "n_steps": 2800,
            "pool_width_mm": round(best["W_mm"], 3),
            "pool_depth_mm": round(best["D_mm"], 3),
            "bead_height_mm": round(best.get("bead_h_mm", 0.0), 3),
            "bead_width_mm": round(best.get("bead_w_mm", 0.0), 3),
            "error_pct_macro": round(best["error_pct"], 2),
        },
        "notes": (
            "Fitted vs jobs/examples/bead_on_plate.yaml reference (macrograph), "
            "full physics stack on standard preset."
        ),
    }
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit bead-on-plate calibration vs macrograph")
    parser.add_argument("--write", action="store_true", help="Write best result to calibration YAML")
    parser.add_argument("--quick", action="store_true", help="Smaller search grid and fewer steps")
    args = parser.parse_args()

    init_taichi(backend="cpu")
    job = load_job_config(_JOB)
    W_ref, D_ref = _macro_reference(job)
    print(f"[fit_calibration] Macro reference W={W_ref} D={D_ref} mm  full physics  preset=standard", flush=True)

    if args.quick:
        best = grid_search(
            [0.78, 0.88],
            [0.9, 1.1],
            [0.9, 1.1],
            [1.0],
            n_steps=1800,
        )
    else:
        best = grid_search(n_steps=2800)

    print(
        f"\n[fit_calibration] BEST η={best['eta']:.3f} σ×={best['sigma_scale']:.3f} "
        f"Ma×={best['marangoni_scale']:.3f} h×={best['heat_loss_factor']:.3f}  "
        f"W={best['W_mm']:.2f} D={best['D_mm']:.2f}  err={best['error_pct']:.1f}%"
    )

    if args.write:
        path = write_calibration_yaml(best)
        print(f"[fit_calibration] Wrote {path}")
    else:
        print("[fit_calibration] Pass --write to update calibration YAML")

    meta_path = pathlib.Path("validation/baselines/calibration_fit_latest.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(best, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
