"""
prediction.py — Locked-physics held-out prediction helpers.

Baseline: jobs/examples/bead_calibrate.yaml
Held-outs may change process I/V/speed/WFS only — Goldak/η/recoil stay frozen.
"""

from __future__ import annotations

import os
from typing import Any

from waam_twin import WAAMTwin
from waam_twin.benchmark import measure_bead_metrics, measure_pool_mm, pool_error_pct
from waam_twin.job import load_job_config
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.validation.bead_helpers import plan_linear_bead_run, run_bead_travel

CALIBRATE_JOB = "jobs/examples/bead_calibrate.yaml"
HELDOUT_FAST_JOB = "jobs/examples/bead_calibrate_heldout_fast.yaml"
HELDOUT_HOT_JOB = "jobs/examples/bead_calibrate_heldout_hot.yaml"
HELDOUT_MACRO2_JOB = "jobs/examples/bead_calibrate_heldout_macro2.yaml"
HELDOUT_BRUNO_JOB = "jobs/examples/bead_bruno_gmaw.yaml"
PIONEER_WALL_JOB = "jobs/examples/wall_pioneer_m1.yaml"

# Keys that must match the calibrate lock exactly for a held-out to be "prediction".
_LOCKED_PROCESS = ("arc_efficiency",)
_LOCKED_GOLDAK = (
    "ff", "fr", "a_front_mm", "a_rear_mm", "b_mm", "c_mm",
)
_LOCKED_ARC = ("sigma_mm", "penetration_mm", "T_vapor_cap_K", "pressure_model")
_LOCKED_ADV = (
    "evap_cooling_scale",
    "T_boiling_K",
    "recoil_accommodation",
    "L_vapor_J_kg",
    "R_spec_vapor_J_kgK",
)


def extract_locked_physics(job: dict[str, Any]) -> dict[str, Any]:
    """Snapshot of fitted knobs that must not change on held-outs."""
    process = job.get("process") or {}
    goldak = job.get("goldak") or {}
    arc = job.get("arc_physics") or {}
    adv = job.get("advanced_physics") or {}
    sim = job.get("simulation") or {}
    return {
        "material": job.get("material"),
        "heat_source": job.get("heat_source"),
        "calibration": job.get("calibration"),
        "physics_tier": sim.get("physics_tier"),
        "enable_recoil": sim.get("enable_recoil"),
        "process": {k: process.get(k) for k in _LOCKED_PROCESS},
        "goldak": {k: goldak.get(k) for k in _LOCKED_GOLDAK},
        "arc_physics": {k: arc.get(k) for k in _LOCKED_ARC},
        "advanced_physics": {k: adv.get(k) for k in _LOCKED_ADV},
    }


def assert_physics_lock(
    baseline_job: dict[str, Any],
    candidate_job: dict[str, Any],
    *,
    label: str = "heldout",
) -> None:
    """Raise if fitted Goldak/η/recoil knobs differ from the calibrate lock."""
    base = extract_locked_physics(baseline_job)
    cand = extract_locked_physics(candidate_job)
    diffs: list[str] = []
    for section in ("material", "heat_source", "calibration", "physics_tier", "enable_recoil"):
        if base[section] != cand[section]:
            diffs.append(f"{section}: {base[section]!r} → {cand[section]!r}")
    for section in ("process", "goldak", "arc_physics", "advanced_physics"):
        for k, v0 in base[section].items():
            v1 = cand[section].get(k)
            if v0 != v1:
                diffs.append(f"{section}.{k}: {v0!r} → {v1!r}")
    if diffs:
        raise AssertionError(
            f"{label} retuned locked physics (not a prediction):\n  " + "\n  ".join(diffs)
        )


def run_job_metrics(
    job_path: str,
    *,
    n_steps: int | None = None,
    equal_distance_m: float | None = None,
    backend: str | None = None,
    reset_backend: bool = True,
) -> dict[str, Any]:
    """Run a bead job and return W/D, crown, T, recoil diagnostics.

    If ``equal_distance_m`` is set, step count follows travel speed so held-outs
    compare the same torch path length (fair heat-input-per-length test).
    """
    if reset_backend:
        reset_taichi()
    init_taichi(backend=backend or os.environ.get("WAAM_BACKEND", "cuda"))
    job = load_job_config(job_path)
    model = job.get("model_reference") or {}
    twin = WAAMTwin.from_job(job_path)
    twin.reset()

    if equal_distance_m is not None and equal_distance_m > 0.0:
        from waam_twin.validation.bead_helpers import steps_for_travel
        n_steps = steps_for_travel(equal_distance_m, twin.travel_speed_m_s, twin.grid.dt)
        max_steps = os.environ.get("WAAM_MAX_BEAD_STEPS")
        if max_steps:
            n_steps = min(n_steps, int(max_steps))
        x_start = max(0.004, 4 * twin.grid.dx)
        y_m = (twin.grid.ny // 2) * twin.grid.dx
        dir_x = 1.0
    else:
        if n_steps is None:
            n_steps = int(os.environ.get("WAAM_BEAD_STEPS", model.get("n_steps", 8000)))
        n_steps, x_start, y_m, dir_x = plan_linear_bead_run(twin, job, n_steps=n_steps)

    run_bead_travel(twin, n_steps, x_start_m=x_start, y_m=y_m, direction_x=dir_x)

    W_mm, D_mm, n_liq = measure_pool_mm(twin)
    bead = measure_bead_metrics(twin)
    telem = twin.get_telemetry()
    fd = telem.get("force_diagnostics") or {}
    process = job.get("process") or {}
    ref = job.get("reference") or {}

    out: dict[str, Any] = {
        "job": job_path,
        "n_steps": n_steps,
        "dt_s": float(twin.grid.dt),
        "travel_distance_mm": float(n_steps * twin.travel_speed_m_s * twin.grid.dt * 1000.0),
        "current_A": float(process.get("current_A", 0)),
        "voltage_V": float(process.get("voltage_V", 0)),
        "travel_mm_s": float(process.get("travel_speed_mm_s", 0)),
        "wire_feed_m_min": float(process.get("wire_feed_m_min", 0)),
        "Q_w_W": float(twin.Q_w),
        "eta": float(twin.eta),
        "recoil_accommodation": float(twin.recoil_accommodation),
        "pool_width_mm": float(W_mm),
        "pool_depth_mm": float(D_mm),
        "n_liquid": int(n_liq),
        "bead_height_mm": float(bead.get("bead_height_mm", telem.get("bead_height_mm", 0))),
        "bead_width_mm": float(bead.get("bead_width_mm", 0)),
        "peak_temp_C": float(telem.get("peak_temp_C", 0)),
        "n_cap": int(telem.get("n_cells_at_vapor_cap", 0)),
        "f_recoil_max": float(fd.get("f_recoil_max", 0)),
        "material_status": str(telem.get("material_status", twin.mat.status)),
        "material_name": str(telem.get("material_name", twin.mat.name)),
        "awaiting_measurement": bool(ref.get("awaiting_measurement", False)),
    }
    W_ref, D_ref = ref.get("pool_width_mm"), ref.get("pool_depth_mm")
    gate = str(ref.get("gate") or "macro_wd")
    out["reference_gate"] = gate
    if not out["awaiting_measurement"] and W_ref is not None and D_ref is not None:
        out["macro_err_pct"] = pool_error_pct(W_mm, D_mm, float(W_ref), float(D_ref))
        out["macro_W_ref"] = float(W_ref)
        out["macro_D_ref"] = float(D_ref)
    # Surface-bead datasets (laser scan): compare external width/height when present.
    Bw_ref, Bh_ref = ref.get("bead_width_mm"), ref.get("bead_height_mm")
    if not out["awaiting_measurement"] and Bw_ref is not None:
        out["bead_W_ref"] = float(Bw_ref)
        bw = max(out["bead_width_mm"], out["pool_width_mm"])
        out["bead_width_err_pct"] = abs(bw - float(Bw_ref)) / max(float(Bw_ref), 1e-6) * 100.0
    if not out["awaiting_measurement"] and Bh_ref is not None:
        out["bead_H_ref"] = float(Bh_ref)
        out["bead_height_err_pct"] = (
            abs(out["bead_height_mm"] - float(Bh_ref)) / max(float(Bh_ref), 1e-6) * 100.0
        )
    if gate == "surface_bead" and out.get("bead_width_err_pct") is not None:
        errs = [out["bead_width_err_pct"]]
        if out.get("bead_height_err_pct") is not None:
            errs.append(out["bead_height_err_pct"])
        out["macro_err_pct"] = max(errs)
        out["macro_W_ref"] = float(Bw_ref) if Bw_ref is not None else float(W_ref or 0)
        out["macro_D_ref"] = float(Bh_ref) if Bh_ref is not None else float("nan")
    return out


def reference_ready(job: dict[str, Any]) -> bool:
    """True when experimental geometry is filled and not awaiting measurement."""
    ref = job.get("reference") or {}
    if ref.get("awaiting_measurement"):
        return False
    gate = str(ref.get("gate") or "macro_wd")
    if gate == "surface_bead":
        return ref.get("bead_width_mm") is not None or ref.get("pool_width_mm") is not None
    return ref.get("pool_width_mm") is not None and ref.get("pool_depth_mm") is not None


def assert_macrograph_prediction(
    metrics: dict[str, Any],
    job: dict[str, Any],
    *,
    threshold_pct: float = 40.0,
    label: str = "heldout",
) -> str:
    """Absolute geometry gate when reference is filled; otherwise PENDING."""
    ref = job.get("reference") or {}
    gate = str(ref.get("gate") or metrics.get("reference_gate") or "macro_wd")
    if not reference_ready(job):
        msg = (
            f"{label}: PENDING macrograph "
            f"(predicted W/D={metrics['pool_width_mm']:.2f}×{metrics['pool_depth_mm']:.2f} mm)"
        )
        print(f"  {msg}")
        return "pending"
    err = float(metrics.get("macro_err_pct", 100.0))
    if gate == "surface_bead":
        print(
            f"  {label}: surface-bead err={err:.1f}% vs "
            f"W_ref={metrics.get('bead_W_ref')} h_ref={metrics.get('bead_H_ref')} mm "
            f"(pred W={metrics['pool_width_mm']:.2f}/{metrics['bead_width_mm']:.2f} "
            f"h={metrics['bead_height_mm']:.2f})"
        )
    else:
        print(
            f"  {label}: macro err={err:.1f}% vs "
            f"{metrics.get('macro_W_ref')}×{metrics.get('macro_D_ref')} mm "
            f"(pred {metrics['pool_width_mm']:.2f}×{metrics['pool_depth_mm']:.2f})"
        )
    if err >= threshold_pct:
        raise AssertionError(
            f"{label} geometry error {err:.1f}% >= {threshold_pct}% — "
            "do not retune Goldak/η/recoil; investigate process match first"
        )
    return "pass"


def assert_trend(baseline: dict[str, Any], case: dict[str, Any], expected: str) -> None:
    """Gate directional physics response without requiring a new macrograph."""
    for label, m in (("baseline", baseline), ("case", case)):
        if m["pool_depth_mm"] < 0.4 or m["n_liquid"] < 30:
            raise AssertionError(
                f"{label} pool not developed enough for trends "
                f"(D={m['pool_depth_mm']:.2f} mm, n_liq={m['n_liquid']}) — "
                f"use full model_reference n_steps (≈8000), not a short smoke budget"
            )
    Wb, Db = baseline["pool_width_mm"], baseline["pool_depth_mm"]
    Wc, Dc = case["pool_width_mm"], case["pool_depth_mm"]
    if expected == "smaller_pool":
        # Grid quantizes W/D to dx (0.4 mm); also accept lower liquid inventory.
        not_grown = (Wc <= Wb + 0.45) and (Dc <= Db + 0.45)
        shrunk_geom = (Wc < Wb - 0.05) or (Dc < Db - 0.05)
        shrunk_liq = case["n_liquid"] < baseline["n_liquid"] * 0.92
        if not (not_grown and (shrunk_geom or shrunk_liq)):
            raise AssertionError(
                f"held-out should shrink pool vs calibrate: "
                f"base W/D={Wb:.2f}/{Db:.2f} n_liq={baseline['n_liquid']}  "
                f"case={Wc:.2f}/{Dc:.2f} n_liq={case['n_liquid']}"
            )
        if case["bead_height_mm"] > baseline["bead_height_mm"] * 1.15:
            raise AssertionError(
                f"held-out crown unexpectedly taller: "
                f"{case['bead_height_mm']:.2f} > {baseline['bead_height_mm']:.2f}"
            )
    elif expected == "larger_pool":
        grown = (Wc > Wb * 1.01) or (Dc > Db * 1.01)
        not_shrunk = (Wc >= Wb * 0.97) and (Dc >= Db * 0.97)
        if not (grown and not_shrunk):
            raise AssertionError(
                f"held-out should enlarge pool vs calibrate: "
                f"base W/D={Wb:.2f}/{Db:.2f}  case={Wc:.2f}/{Dc:.2f}"
            )
        if case["peak_temp_C"] < baseline["peak_temp_C"] - 50.0:
            raise AssertionError(
                f"held-out T_peak should not drop: "
                f"{case['peak_temp_C']:.0f}C < {baseline['peak_temp_C']:.0f}C"
            )
    else:
        raise ValueError(f"unknown expected_trend={expected!r}")
