# Thermal validation report template (v1)

## Run metadata

| Field | Value |
|-------|-------|
| Date | |
| Commit | |
| Preset | |
| Material | |
| Job file | |
| Backend | |

## Kernel-only benchmarks

| Test | Metric | Threshold | Result |
|------|--------|-----------|--------|
| `thermal_diffusion` | L2 vs Gaussian | < 12% | |
| `stefan_solidification` | Front √t slope | < 15% | |
| `mass_conservation` | ρ drift | < 2% | |
| `lbm_poiseuille` | Profile L2 | < 15% | |

## Process benchmarks (traveling arc, minimal preset)

| Test | Metric | Threshold | Result |
|------|--------|-----------|--------|
| `rosenthal_farfield` | Tail T vs 2D Rosenthal | < 55% | |
| `thermocouple` | Probe T vs Rosenthal @ 3 mm | < 40% | |
| `pool_geometry` | W/D vs scaled reference | < 70% | |
| `soak_10k` | 10k steps, no NaN, ρ drift | < 2% | |

Tighter ±20% macrograph targets require `standard`/`high` preset (finer dx).

## Phase 2 (CFD / VOF)

| Test | Metric | Threshold | Result |
|------|--------|-----------|--------|
| `vof_mass` | φ sum drift | < 2% | |

## Notes

Document material status (`placeholder` vs `calibrated`), calibration file, and any deviations from reference process parameters.
