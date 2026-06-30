"""
test_two_layer_haz_ref.py — 2-layer HAZ peak T vs job reference band.

Qualitative comparison to literature/FEA envelope for ER70S-6 WAAM interpass
(peak HAZ typically 600–3000 K depending on spacing; not full FEA parity).
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin


def run(n_steps: int = 600) -> float:
    init_taichi(backend="cpu")
    job_path = "jobs/examples/two_layer.yaml"
    twin = WAAMTwin.from_job(job_path)
    ref = twin._job_config.get("reference", {}) if hasattr(twin, "_job_config") else {}
    t_min = float(ref.get("haz_T_peak_min_K", 600))
    t_max = float(ref.get("haz_T_peak_max_K", 3200))

    twin.enable_vof = False
    twin.enable_heat_loss = True
    twin.enable_substrate_growth = True
    twin.reset()
    twin.run_path(job_path, n_steps=n_steps, interpass_steps=80)

    T_max = twin.grid.T_max.to_numpy()
    flags = twin.grid.flags.to_numpy()
    metal = flags != twin.grid.FLAG_GAS
    peak = float(T_max[metal].max()) if metal.any() else float(T_max.max())

    print(f"[two_layer_haz_ref] T_max peak={peak:.0f}K  band=[{t_min:.0f}, {t_max:.0f}]K")
    if peak < t_min:
        raise AssertionError(f"HAZ peak {peak:.0f}K below reference min {t_min:.0f}K")
    if peak > t_max:
        raise AssertionError(f"HAZ peak {peak:.0f}K above reference max {t_max:.0f}K")
    return peak


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
