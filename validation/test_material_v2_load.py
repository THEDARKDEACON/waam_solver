"""
test_material_v2_load.py — ER70S-6.v2 loads with richer tables than v1.

Does not switch the calibrate job. Ensures the v2 property path is ready
for a future re-lock after FULL + prediction_report.
"""

from __future__ import annotations

import sys

from waam_twin.materials import load_material


def run() -> None:
    v1 = load_material("materials/validated/ER70S-6.v1.yaml")
    v2 = load_material("materials/validated/ER70S-6.v2.yaml")
    if v1.status != "calibrated" or v2.status != "calibrated":
        raise AssertionError(f"expected calibrated status, got v1={v1.status} v2={v2.status}")
    n1 = len(v1.tables.cp) if v1.tables and v1.tables.cp else 0
    n2 = len(v2.tables.cp) if v2.tables and v2.tables.cp else 0
    print(
        f"[material_v2] v1 cp_knots={n1}  v2 cp_knots={n2}  "
        f"v2.k(300)={v2.k_at(300):.1f}  v2.k(1793)={v2.k_at(1793):.1f}  "
        f"notes={'yes' if getattr(v2, 'notes', '') else 'no'}"
    )
    if n2 <= n1:
        raise AssertionError(f"v2 should have denser cp table ({n2} <= {n1})")
    if abs(v2.T_liquidus - v1.T_liquidus) > 1e-6:
        raise AssertionError("v2 liquidus should match v1 process window")
    # Solid k should exceed liquid-ish knot near melting for mild steel trend.
    if v2.k_at(300.0) <= v2.k_at(1793.0):
        raise AssertionError("v2 k(T) should be higher at ambient than near liquidus")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
