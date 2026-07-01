"""
test_surfactant_dgamma.py — Surfactant ppm flips or scales dγ/dT sign.
"""

from __future__ import annotations

from waam_twin.materials import load_material
from waam_twin.physics.surfactant import effective_dgamma_dT, surfactant_dgamma_dT_scale


def run() -> None:
    base = -4.3e-4
    low = effective_dgamma_dT(base, 20.0)
    high = effective_dgamma_dT(base, 80.0)
    assert low < 0, "low-S steel should keep outward (negative) dγ/dT"
    assert high > 0, "high-S steel should flip to inward (positive) dγ/dT"
    assert surfactant_dgamma_dT_scale(25.0) == 1.0
    print(f"[surfactant] low-S dγ/dT={low:.2e}  high-S dγ/dT={high:.2e}")

    mat = load_material("materials/validated/ER70S-6.v1.yaml")
    assert mat.dgamma_dT < 0
    assert mat.sulphur_ppm <= 35.0
    print(f"[surfactant] ER70S-6 loaded dγ/dT={mat.dgamma_dT:.2e}  S={mat.sulphur_ppm} ppm")


if __name__ == "__main__":
    run()
    print("PASS")
