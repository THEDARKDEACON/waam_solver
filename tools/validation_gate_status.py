"""
validation_gate_status.py — Report which experimental gates are closed / open.

Does not run long GPU sims. Checks job YAML references + physics locks.

Usage:
  PYTHONPATH=. python3 -m waam_twin.tools.validation_gate_status
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from waam_twin.job import load_job_config
from waam_twin.validation.prediction import (
    CALIBRATE_JOB,
    HELDOUT_BRUNO_JOB,
    HELDOUT_FAST_JOB,
    HELDOUT_HOT_JOB,
    HELDOUT_MACRO2_JOB,
    PIONEER_WALL_JOB,
    assert_physics_lock,
    reference_ready,
)

TWOLAYER_JOB = "jobs/examples/bead_calibrate_twolayer.yaml"
FH30_JOB = "jobs/examples/bead_crown_fh30.yaml"


def _status(job_path: str) -> dict:
    job = load_job_config(job_path)
    ref = job.get("reference") or {}
    awaiting = bool(ref.get("awaiting_measurement", False))
    gate = str(ref.get("gate") or "macro_wd")
    ready = reference_ready(job) if gate in ("macro_wd", "surface_bead") else (
        not awaiting and (
            ref.get("wall_width_mm") is not None
            or ref.get("remelt_depth_mm") is not None
            or (ref.get("pool_width_mm") is not None and ref.get("pool_depth_mm") is not None)
        )
    )
    return {
        "job": job_path,
        "material": job.get("material"),
        "gate": gate,
        "awaiting_measurement": awaiting,
        "reference_ready": bool(ready),
        "pool_width_mm": ref.get("pool_width_mm"),
        "pool_depth_mm": ref.get("pool_depth_mm"),
        "bead_width_mm": ref.get("bead_width_mm"),
        "bead_height_mm": ref.get("bead_height_mm"),
        "wall_width_mm": ref.get("wall_width_mm"),
        "remelt_depth_mm": ref.get("remelt_depth_mm"),
        "haz_width_mm": ref.get("haz_width_mm"),
        "source": ref.get("source"),
        "material_match": ref.get("material_match"),
    }


def main() -> int:
    base = load_job_config(CALIBRATE_JOB)
    rows = []
    lock_ok = True
    for path, label in (
        (CALIBRATE_JOB, "calibrate_ER70S6"),
        (HELDOUT_FAST_JOB, "heldout_fast"),
        (HELDOUT_HOT_JOB, "heldout_hot"),
        (HELDOUT_MACRO2_JOB, "heldout_macro2"),
        (HELDOUT_BRUNO_JOB, "heldout_bruno"),
        (PIONEER_WALL_JOB, "pioneer_m1_wall"),
        (TWOLAYER_JOB, "twolayer_calibrate"),
        (FH30_JOB, "crown_fh30"),
    ):
        try:
            st = _status(path)
            st["label"] = label
            if path != CALIBRATE_JOB and path != FH30_JOB:
                try:
                    assert_physics_lock(base, load_job_config(path), label=label)
                    st["physics_lock"] = "ok"
                except AssertionError as exc:
                    st["physics_lock"] = f"FAIL: {exc}"
                    lock_ok = False
            elif path == FH30_JOB:
                st["physics_lock"] = "n/a (different alloy track)"
            else:
                st["physics_lock"] = "baseline"
            rows.append(st)
        except Exception as exc:
            rows.append({"label": label, "job": path, "error": str(exc)})
            lock_ok = False

    print("=== Validation gate status ===\n")
    closed = open_ = 0
    for r in rows:
        if r.get("error"):
            print(f"  FAIL  {r['label']}: {r['error']}")
            open_ += 1
            continue
        ready = r.get("reference_ready")
        mark = "CLOSED" if ready else "OPEN  "
        if ready:
            closed += 1
        else:
            open_ += 1
        detail = []
        for k in ("pool_width_mm", "pool_depth_mm", "bead_width_mm", "bead_height_mm",
                  "wall_width_mm", "remelt_depth_mm"):
            if r.get(k) is not None:
                detail.append(f"{k}={r[k]}")
        print(
            f"  {mark}  {r['label']:22s}  lock={r.get('physics_lock')}  "
            f"gate={r.get('gate')}  " + (" ".join(detail) or "(no ref numbers)")
        )
        if r.get("source"):
            print(f"           source: {r['source']}")

    print(f"\nSummary: {closed} closed / {open_} open  (physics locks {'OK' if lock_ok else 'FAIL'})")
    print(
        "\nNext blockers:\n"
        "  - heldout_macro2: need ER70S-6 macro at 5 mm/s (or change slot process to match a coupon)\n"
        "  - twolayer_calibrate: need remelt/HAZ at calibrate process (100 A / 6.5 mm/s)\n"
        "  - crown_fh30: need FH-30 coupon + promote material placeholder → validated\n"
        "  - GPU: prediction_report --with-bruno ; multipass_report --job wall_pioneer_m1.yaml\n"
        "  - Do not retune Goldak / η / evap_cooling_scale / C_acc — held-out jobs lock them.\n"
        "    Tighten pool tripwires with WAAM_POOL_GATE_PCT instead."
    )

    out = Path("docs/validation/data/gate_status_latest.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"gates": rows, "closed": closed, "open": open_}, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
    except OSError:
        pass
    return 0 if lock_ok else 1


if __name__ == "__main__":
    sys.exit(main())
