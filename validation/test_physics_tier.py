"""
test_physics_tier.py — physics_tier: full enables Lorentz, gas shear, Lin–Eagar.
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.job import apply_physics_tier, apply_job_to_twin
from waam_twin.physics.weld_forces import arc_pressure_peak_pa


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=16, ny=12, nz=12, dx=3e-4, max_tracers=5, welding_current_A=180.0)
    twin.reset()

    apply_physics_tier(twin, "full")
    if not twin.enable_lorentz:
        raise AssertionError("physics_tier=full must enable Lorentz")
    if not twin.enable_gas_shear:
        raise AssertionError("physics_tier=full must enable gas shear")
    if twin.arc_pressure_model != "lin_eagar":
        raise AssertionError("physics_tier=full must set arc_pressure_model=lin_eagar")
    if not twin.grid._has_lorentz:
        raise AssertionError("Lorentz fields must be allocated")

    p = arc_pressure_peak_pa(twin)
    if p <= 0.0:
        raise AssertionError(f"Lin–Eagar peak pressure must be > 0, got {p}")

    # Explicit override still wins after tier
    apply_job_to_twin(twin, {
        "simulation": {"physics_tier": "full", "enable_lorentz": False},
        "process": {"current_A": 180},
    })
    if twin.enable_lorentz:
        raise AssertionError("explicit enable_lorentz: false must override tier")

    print(f"[physics_tier] full OK  p_arc={p:.1f} Pa  override OK")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
