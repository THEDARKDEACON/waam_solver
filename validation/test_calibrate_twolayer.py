"""
test_calibrate_twolayer.py — Remelt + HAZ on calibrate-domain two-layer job.

Uses bead_calibrate_twolayer.yaml (Goldak / full tier). Asserts liquid appears
during the path and HAZ peak sits in the job reference band.
"""

from __future__ import annotations

import os
import sys

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin
from waam_twin.job import load_job_config


def run(n_steps: int | None = None) -> float:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cuda"))
    job_path = "jobs/examples/bead_calibrate_twolayer.yaml"
    job = load_job_config(job_path)
    model = job.get("model_reference") or {}
    ref = job.get("reference") or {}
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_TWOLAYER_STEPS", model.get("n_steps", 12000)))
    interpass = int(model.get("interpass_steps", 200))

    twin = WAAMTwin.from_job(job_path)
    twin.enable_vof = True
    twin.enable_substrate_growth = True
    twin.use_torch_z = True
    twin.reset()
    twin.run_path(job_path, n_steps=n_steps, interpass_steps=interpass)

    fl = int((twin.grid.f_l.to_numpy() > 0.5).sum())
    T_max = twin.grid.T_max.to_numpy()
    flags = twin.grid.flags.to_numpy()
    metal = flags != twin.grid.FLAG_GAS
    peak = float(T_max[metal].max()) if metal.any() else float(T_max.max())

    t_min = float(ref.get("haz_T_peak_min_K", 800))
    t_max = float(ref.get("haz_T_peak_max_K", 3200))
    remelt_min = int(ref.get("remelt_liquid_min", 5))

    print(
        f"[calibrate_twolayer] liquid={fl}  HAZ={peak:.0f}K  "
        f"band=[{t_min:.0f},{t_max:.0f}]  steps={n_steps}"
    )
    if fl < remelt_min:
        raise AssertionError(f"insufficient liquid cells {fl} < {remelt_min}")
    if peak < t_min or peak > t_max:
        raise AssertionError(f"HAZ peak {peak:.0f}K outside [{t_min:.0f},{t_max:.0f}]")
    return peak


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
