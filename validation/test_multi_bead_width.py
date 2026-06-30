"""
test_multi_bead_width.py — Bead path length smoke (CSV path + width telemetry).
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.job import load_job_config


def run(n_steps: int = 700, threshold_pct: float = 15.0) -> float:
    init_taichi(backend="cpu")
    job = load_job_config("jobs/examples/multi_bead.yaml")
    job["torch_path_csv"] = "jobs/paths/bead_line.csv"
    job.pop("torch_path", None)

    twin = WAAMTwin.from_job("jobs/examples/multi_bead.yaml")
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.reset()
    twin.run_path(job, n_steps=n_steps)

    telem = twin.get_telemetry()
    w = telem["pool_width_mm"]
    model_w = float(job.get("model_reference", {}).get("pool_width_mm", 2.5))
    err = abs(w - model_w) / max(model_w, 0.1) * 100.0
    ref = float(job.get("reference", {}).get("pool_width_mm", 7.0))
    err_macro = abs(w - ref) / ref * 100.0 if ref > 0 else 0.0

    print(
        f"[multi_bead_width] W={w:.2f}mm  model={model_w:.1f}mm err={err:.1f}%  "
        f"macro={ref:.1f}mm err={err_macro:.1f}%  (threshold {threshold_pct}%)"
    )
    if err >= threshold_pct:
        raise AssertionError(f"Bead width error {err:.1f}% >= {threshold_pct}%")
    return w


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
