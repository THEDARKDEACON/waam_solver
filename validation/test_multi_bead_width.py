"""
test_multi_bead_width.py — Bead path length smoke (CSV/path + width telemetry).
"""

from __future__ import annotations

import sys

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin.job import load_job_config


def run(n_steps: int = 4000, threshold_pct: float = 50.0) -> float:
    init_taichi(backend="cpu")
    job_path = "jobs/examples/multi_bead.yaml"
    job = load_job_config(job_path)
    # Neutralize η-reducing calibration so a short CI budget can melt.
    job["calibration"] = None

    twin = WAAMTwin.from_job(job_path)
    twin.eta = float(job["process"].get("arc_efficiency", 0.72))
    twin.sigma_cells = twin.arc_sigma_m / twin.grid.dx  # undo cal σ scale if applied
    twin.enable_vof = False
    twin.enable_heat_loss = False
    twin.enable_recoil = False
    twin.enable_lorentz = False
    twin.enable_gas_shear = False
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
        f"macro={ref:.1f}mm err={err_macro:.1f}%  (threshold {threshold_pct}%)  "
        f"T_peak={telem['peak_temp_C']:.0f}C"
    )
    if w < 0.5:
        raise AssertionError("Bead width smoke produced no melt pool")
    if err >= threshold_pct and abs(w - model_w) > 2.0:
        raise AssertionError(f"Bead width error {err:.1f}% >= {threshold_pct}%")
    return w


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
