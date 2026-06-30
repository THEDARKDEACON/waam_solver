"""
fit_calibration.py — Grid-search η and arc_sigma_scale vs bead-on-plate reference W/D.

Usage:
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.fit_calibration
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.tools.fit_calibration --write
"""

from __future__ import annotations

import argparse
import json
import pathlib

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_pool_mm, pool_error_pct
from waam_twin.calibration import CalibrationProfile, load_calibration
from waam_twin.job import apply_job_to_twin, load_job_config
from waam_twin.platform import init_taichi
from waam_twin.validation.torch_motion import torch_position_linear

_JOB = "jobs/examples/bead_on_plate.yaml"
_CAL_OUT = "materials/calibration/ER70S-6.bead_on_plate.yaml"


def _reference_wd(job: dict) -> tuple[float, float, str]:
    """Return (W_ref, D_ref, ref_kind) preferring model_reference when present."""
    model = job.get("model_reference", {})
    if model.get("pool_width_mm") is not None:
        return (
            float(model["pool_width_mm"]),
            float(model.get("pool_depth_mm", 2.8)),
            "model",
        )
    ref = job.get("reference", {})
    return (
        float(ref.get("pool_width_mm", 7.0)),
        float(ref.get("pool_depth_mm", 2.8)),
        "macro",
    )


def run_benchmark(
    eta: float,
    sigma_scale: float,
    heat_loss_factor: float = 1.0,
    n_steps: int = 2000,
) -> tuple[float, float, float]:
    """Return (W_mm, D_mm, error_pct) at standard dx benchmark grid."""
    job = load_job_config(_JOB)
    W_ref, D_ref, _ = _reference_wd(job)
    proc = job["process"]
    travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
    arc_w = float(proc["current_A"]) * float(proc["voltage_V"])

    twin = WAAMTwin(
        material=job["material"],
        nx=88, ny=44, nz=44,
        dx=3.0e-4,
        arc_power_W=arc_w,
        arc_efficiency=eta,
        enable_heat_loss=False,
        max_tracers=100,
    )
    apply_job_to_twin(twin, job)
    twin.enable_vof = False
    twin.sigma_cells *= sigma_scale
    twin.h_conv *= heat_loss_factor
    twin.travel_speed_m_s = travel
    twin.reset()

    g = twin.grid
    cy = (g.ny // 2) * g.dx
    x_start = 0.015
    for step in range(n_steps):
        x, _ = torch_position_linear(step, g.dt, travel, x_start, cy)
        twin.step(x, cy, is_welding=True)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    if n_liq < 5:
        return W_mm, D_mm, 999.0
    err = pool_error_pct(W_mm, D_mm, W_ref, D_ref)
    return W_mm, D_mm, err


def grid_search(
    eta_vals: list[float] | None = None,
    sigma_vals: list[float] | None = None,
) -> dict:
    eta_vals = eta_vals or [0.65, 0.72, 0.80, 0.85, 0.92, 1.0]
    sigma_vals = sigma_vals or [0.6, 0.8, 1.0, 1.2, 1.4]
    best = {"error_pct": 1e9, "eta": 0.72, "sigma_scale": 1.0, "W_mm": 0.0, "D_mm": 0.0}

    for eta in eta_vals:
        for sig in sigma_vals:
            W, D, err = run_benchmark(eta, sig)
            print(f"  η={eta:.2f} σ×={sig:.2f}  W={W:.2f} D={D:.2f}  err={err:.1f}%")
            if err < best["error_pct"]:
                best = {
                    "error_pct": err,
                    "eta": eta,
                    "sigma_scale": sig,
                    "W_mm": W,
                    "D_mm": D,
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
        "process": "bead_on_plate_reference",
        "arc_efficiency": round(best["eta"], 4),
        "heat_loss_factor": 1.0,
        "marangoni_scale": 1.0,
        "arc_sigma_scale": round(best["sigma_scale"], 4),
        "fit_metrics": {
            "grid": [88, 44, 44],
            "dx_mm": 0.3,
            "n_steps": 2000,
            "pool_width_mm": round(best["W_mm"], 3),
            "pool_depth_mm": round(best["D_mm"], 3),
            "error_pct_model": round(best["error_pct"], 2),
        },
        "notes": (
            "Auto-fitted by waam_twin.tools.fit_calibration on 88×44×44 @ dx=0.3 mm. "
            "error_pct_model is vs jobs/examples/bead_on_plate.yaml model_reference."
        ),
    }
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit bead-on-plate calibration scalars")
    parser.add_argument("--write", action="store_true", help="Write best result to calibration YAML")
    parser.add_argument("--quick", action="store_true", help="Smaller search grid")
    args = parser.parse_args()

    init_taichi(backend="cpu")
    job = load_job_config(_JOB)
    _, _, ref_kind = _reference_wd(job)
    print(f"[fit_calibration] Grid search on 88×44×44 @ dx=0.3mm  ref={ref_kind} …")
    if args.quick:
        best = grid_search([0.72, 0.85, 0.95], [0.8, 1.0, 1.2])
    else:
        best = grid_search()

    print(
        f"\n[fit_calibration] BEST η={best['eta']:.3f} σ×={best['sigma_scale']:.3f}  "
        f"W={best['W_mm']:.2f}mm D={best['D_mm']:.2f}mm  err={best['error_pct']:.1f}%"
    )

    if args.write:
        path = write_calibration_yaml(best)
        print(f"[fit_calibration] Wrote {path}")
    else:
        print("[fit_calibration] Pass --write to update calibration YAML")

    meta_path = pathlib.Path("waam_twin/validation/baselines/calibration_fit_latest.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(best, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
