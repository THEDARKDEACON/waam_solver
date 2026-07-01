"""
run_all.py — Validation entry point.

Usage:
    WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all

Set WAAM_FULL_VALIDATION=1 for process benchmarks + soak.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("WAAM_BACKEND", "cpu")
    os.environ.setdefault("WAAM_PRESET", "minimal")

    tests = [
        ("thermal_diffusion", "waam_twin.validation.test_thermal_diffusion"),
        ("mass_conservation", "waam_twin.validation.test_mass_conservation"),
        ("lbm_poiseuille", "waam_twin.validation.test_lbm_poiseuille"),
        ("lbm_cavity", "waam_twin.validation.test_lbm_cavity"),
        ("stefan_solidification", "waam_twin.validation.test_stefan_solidification"),
        ("vof_mass", "waam_twin.validation.test_vof_mass"),
        ("laplace_csf", "waam_twin.validation.test_laplace"),
        ("marangoni_cell", "waam_twin.validation.test_marangoni_cell"),
        ("heated_cavity", "waam_twin.validation.test_heated_cavity"),
        ("multi_bead", "waam_twin.validation.test_multi_bead"),
        ("arc_pressure", "waam_twin.validation.test_arc_pressure"),
        ("moving_window", "waam_twin.validation.test_moving_window"),
        ("surface_vtk", "waam_twin.validation.test_surface_vtk"),
        ("viewer_session", "waam_twin.validation.test_viewer_session"),
        ("viewer_extract", "waam_twin.validation.test_viewer_extract"),
        ("export_full_vtk", "waam_twin.validation.test_export_full_vtk"),
        ("export_bundle", "waam_twin.validation.test_export_bundle"),
        ("probe_recorder", "waam_twin.validation.test_probe_recorder"),
        ("enthalpy_cap", "waam_twin.validation.test_enthalpy_cap"),
        ("mass_balance", "waam_twin.validation.test_mass_balance"),
        ("arc_surface_weight", "waam_twin.validation.test_arc_surface_weight"),
        ("advanced_weld_forces", "waam_twin.validation.test_advanced_weld_forces"),
        ("wetting_droplet", "waam_twin.validation.test_wetting_droplet"),
        ("deposition_no_column", "waam_twin.validation.test_deposition_no_column"),
        ("hydrostatic_gravity", "waam_twin.validation.test_hydrostatic_gravity"),
        ("bead_freeze", "waam_twin.validation.test_bead_freeze"),
        ("stickout_preheat", "waam_twin.validation.test_stickout_preheat"),
        ("backend_smoke", "waam_twin.validation.test_backend_smoke"),
    ]

    if os.environ.get("WAAM_FULL_VALIDATION") == "1":
        tests.extend([
            ("rosenthal_farfield", "waam_twin.validation.test_rosenthal_farfield"),
            ("thermocouple", "waam_twin.validation.test_thermocouple"),
            ("pool_geometry", "waam_twin.validation.test_pool_geometry"),
            ("job_parity", "waam_twin.validation.test_job_parity"),
            ("soak_10k", "waam_twin.validation.test_soak_10k"),
            ("interpass_haz", "waam_twin.validation.test_interpass_haz"),
            ("parametric_monotonic", "waam_twin.validation.test_parametric_monotonic"),
            ("multi_bead_width", "waam_twin.validation.test_multi_bead_width"),
            ("calibrated_pool", "waam_twin.validation.test_calibrated_pool"),
            ("two_layer_remelt", "waam_twin.validation.test_two_layer_remelt"),
            ("two_layer_haz_ref", "waam_twin.validation.test_two_layer_haz_ref"),
        ])

    if os.environ.get("WAAM_STANDARD_VALIDATION") == "1":
        tests.append(("pool_geometry_standard", "waam_twin.validation.test_pool_geometry_standard"))

    failed = []
    for name, module in tests:
        print(f"\n{'='*60}\n  Running: {name}\n{'='*60}")
        try:
            mod = __import__(module, fromlist=["run"])
            mod.run()
            print(f"  ✅ {name}")
        except Exception as exc:
            print(f"  ❌ {name}: {exc}")
            failed.append(name)

    if failed:
        print(f"\nFailed: {failed}")
        return 1
    print("\n🏁 All validation tests passed.")
    if os.environ.get("WAAM_FULL_VALIDATION") != "1":
        print("  (Full suite skipped — set WAAM_FULL_VALIDATION=1)")
    if os.environ.get("WAAM_STANDARD_VALIDATION") != "1":
        print("  (Standard preset pool test skipped — set WAAM_STANDARD_VALIDATION=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
