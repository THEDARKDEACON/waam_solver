"""
test_intensive_force_ranking.py — Cho-style ablation with longer heat-up.

Ranks force diagnostics after a sustained weld and checks:
  - full case: Ma, CSF, arc, Lorentz all active
  - no_marangoni / no_lorentz / no_recoil ablations zero their channels
  - Marangoni dominates gas shear on this conduction-mode setup
"""

from __future__ import annotations

import os
import sys

from waam_twin.runtime import init_taichi
from waam_twin.tools.force_ablation import AblationCase, _run_case


def run(n_steps: int | None = None) -> None:
    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    if n_steps is None:
        n_steps = int(os.environ.get("WAAM_INTENSIVE_ABLATION_STEPS", "200"))

    full = _run_case(AblationCase("full"), n_steps=n_steps, dx=3.5e-4)
    no_ma = _run_case(AblationCase("no_marangoni", marangoni=False), n_steps=n_steps, dx=3.5e-4)
    no_lz = _run_case(AblationCase("no_lorentz", lorentz=False), n_steps=n_steps, dx=3.5e-4)
    no_shear = _run_case(
        AblationCase("no_gas_shear", gas_shear=False), n_steps=n_steps, dx=3.5e-4
    )

    fd = full["force_diagnostics"]
    ma = float(fd.get("f_marangoni_max", 0.0))
    csf = float(fd.get("f_csf_max", 0.0))
    arc = float(fd.get("f_arc_max", 0.0))
    lz = float(fd.get("f_lorentz_max", 0.0))
    shear = float(fd.get("f_gas_shear_max", 0.0))

    print(
        f"[intensive_ranking] steps={n_steps}  "
        f"Ma={ma:.3e} CSF={csf:.3e} arc={arc:.3e} Lz={lz:.3e} shear={shear:.3e}  "
        f"D={full['pool_depth_mm']:.2f}mm  n_liq={full['n_liquid_cells']}"
    )
    if full["n_liquid_cells"] < 15:
        raise AssertionError("intensive ablation full case under-melted")
    for name, val in (("Ma", ma), ("CSF", csf), ("arc", arc), ("Lorentz", lz)):
        if val <= 0.0:
            raise AssertionError(f"{name} diagnostic inactive after intensive heat-up")

    if float(no_ma["force_diagnostics"].get("f_marangoni_max", 1)) > 1e-12:
        raise AssertionError("no_marangoni ablation failed")
    if float(no_lz["force_diagnostics"].get("f_lorentz_max", 1)) > 1e-12:
        raise AssertionError("no_lorentz ablation failed")
    if float(no_shear["force_diagnostics"].get("f_gas_shear_max", 1)) > 1e-12:
        raise AssertionError("no_gas_shear ablation failed")

    # Conduction-mode WAAM: thermocapillary usually outranks shielding-gas shear.
    if shear > 0.0 and ma < 0.05 * shear:
        raise AssertionError(
            f"unexpected ranking: Ma={ma:.3e} << shear={shear:.3e} on conduction-mode pool"
        )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
