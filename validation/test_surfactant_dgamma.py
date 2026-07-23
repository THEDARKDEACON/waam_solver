"""
test_surfactant_dgamma.py — Heiple scale + Sahoo/DebRoy γ(T,a_S).

PHYSICS_FORCE_CORRECTNESS_SPEC §5.9 / WP-H.
"""

from __future__ import annotations

from waam_twin.materials import load_material
from waam_twin.physics.surfactant import (
    dgamma_dT_sahoo,
    effective_dgamma_dT,
    gamma_sahoo,
    surfactant_dgamma_dT_scale,
)


def run() -> None:
    # --- Heiple static scale ---
    base = -4.3e-4
    low = effective_dgamma_dT(base, 20.0)
    high = effective_dgamma_dT(base, 80.0)
    assert low < 0, "low-S steel should keep outward (negative) dγ/dT"
    assert high > 0, "high-S steel should flip to inward (positive) dγ/dT"
    assert surfactant_dgamma_dT_scale(25.0) == 1.0
    print(f"[surfactant] Heiple low-S dγ/dT={low:.2e}  high-S dγ/dT={high:.2e}")

    # --- Sahoo/DebRoy analytic ---
    T_near = 1850.0
    dg_low = dgamma_dT_sahoo(T_near, 20.0)
    dg_high = dgamma_dT_sahoo(T_near, 150.0)
    dg_hot_high = dgamma_dT_sahoo(2500.0, 150.0)
    assert dg_low < 0, f"Sahoo low-S near liquidus should be negative, got {dg_low}"
    assert dg_high > 0, (
        f"Sahoo high-S near liquidus should be positive (inward), got {dg_high}"
    )
    # Classic Sahoo: at high T the coefficient returns toward negative
    assert dg_hot_high < dg_high, "Sahoo dγ/dT should decrease toward high T at fixed a_S"
    g_pure = gamma_sahoo(1809.0, 0.0)
    assert abs(g_pure - 1.943) < 1e-6, f"pure Fe γ(Tm) expected 1.943, got {g_pure}"
    print(
        f"[surfactant] Sahoo dγ/dT(1850K): low-S={dg_low:.2e}  high-S={dg_high:.2e}  "
        f"high-S@2500K={dg_hot_high:.2e}"
    )

    # --- Material YAML with model: sahoo ---
    mat = load_material("materials/validated/ER70S-6.v1.yaml")
    assert mat.sulphur_ppm <= 35.0
    assert mat.dgamma_dT < 0, "calibrated low-S ER70S-6 should stay outward near liquidus"
    assert len(mat.tables.dgamma_dT) >= 5, "Sahoo should install a dγ/dT table"
    # Table must be T-local (not a constant scale of the YAML two-point table)
    dg_vals = [v for _, v in mat.tables.dgamma_dT]
    assert min(dg_vals) != max(dg_vals), "Sahoo table must vary with T"
    print(
        f"[surfactant] ER70S-6(sahoo) dγ/dT={mat.dgamma_dT:.2e}  "
        f"S={mat.sulphur_ppm} ppm  N_knots={len(mat.tables.dgamma_dT)}  "
        f"γ0={mat.gamma_0:.3f}"
    )


if __name__ == "__main__":
    run()
    print("PASS")
