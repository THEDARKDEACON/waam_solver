"""
test_force_ablation.py — Cho & Na-style: forces present and ranking smoke.

Asserts:
  - full tier produces nonzero CSF / Marangoni / arc diagnostics after heat-up
  - disabling Marangoni zeros f_marangoni_max
  - Lorentz on contributes nonzero f_lorentz_max when enabled
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin.tools.force_ablation import AblationCase, _run_case


def run() -> None:
    init_taichi(backend="cpu")
    full = _run_case(AblationCase("full"), n_steps=40, dx=3.5e-4)
    no_ma = _run_case(AblationCase("no_marangoni", marangoni=False), n_steps=40, dx=3.5e-4)
    no_lz = _run_case(AblationCase("no_lorentz", lorentz=False), n_steps=40, dx=3.5e-4)

    fd = full["force_diagnostics"]
    print(
        f"[force_ablation] full D={full['pool_depth_mm']:.3f}  "
        f"Ma={fd.get('f_marangoni_max', 0):.3e}  "
        f"CSF={fd.get('f_csf_max', 0):.3e}  "
        f"arc={fd.get('f_arc_max', 0):.3e}  "
        f"Lz={fd.get('f_lorentz_max', 0):.3e}"
    )
    if full["n_liquid_cells"] < 5:
        raise AssertionError("full ablation case produced almost no liquid")
    if fd.get("f_csf_max", 0) <= 0:
        raise AssertionError("CSF diagnostic is zero — surface tension inactive")
    if fd.get("f_marangoni_max", 0) <= 0:
        raise AssertionError("Marangoni diagnostic is zero on heated pool")
    if fd.get("f_arc_max", 0) <= 0:
        raise AssertionError("Arc pressure diagnostic is zero")
    if fd.get("f_lorentz_max", 0) <= 0:
        raise AssertionError("Lorentz diagnostic is zero with enable_lorentz")

    fd_ma = no_ma["force_diagnostics"]
    if fd_ma.get("f_marangoni_max", 1) > 1e-12:
        raise AssertionError(
            f"no_marangoni should zero Ma force, got {fd_ma.get('f_marangoni_max')}"
        )

    fd_lz = no_lz["force_diagnostics"]
    if fd_lz.get("f_lorentz_max", 1) > 1e-12:
        raise AssertionError(
            f"no_lorentz should zero Lorentz force, got {fd_lz.get('f_lorentz_max')}"
        )
    print(
        f"[force_ablation] no_Ma Ma={fd_ma.get('f_marangoni_max', 0):.3e}  "
        f"no_Lz Lz={fd_lz.get('f_lorentz_max', 0):.3e}"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
