# Reference case: ER70S-6 bead-on-plate

## Experimental / literature reference

| Quantity | Macrograph value | Source |
|----------|------------------|--------|
| Pool width | 7.0 mm | `jobs/examples/bead_on_plate.yaml` → `reference` |
| Pool depth | 2.8 mm | Same |
| Arc power | 2.8 kW (140 A × 20 V) | Job process block |
| Travel speed | 5 mm/s | Job process block |

## Simulator envelope (`model_reference`)

On **88×44×44** cells at **dx = 0.3 mm**, Gaussian2D heat source, calibrated η=0.65 and σ×=0.8:

| Quantity | Model value | Calibration file |
|----------|-------------|------------------|
| Pool width | 4.2 mm | `fit_metrics` in `ER70S-6.bead_on_plate.yaml` |
| Pool depth | 5.4 mm | Same |
| Steps | 2000 | Traveling arc from x = 15 mm |

Measured reproduction error vs `fit_metrics`: **~11%** (`test_calibrated_pool`).

## Comparison to macrograph

| Metric | Macrograph | Best model | Error |
|--------|------------|------------|-------|
| Width | 7.0 mm | ~3.9–4.2 mm | ~40–45% |
| Depth | 2.8 mm | ~5.4–6.0 mm | ~90% |

**Conclusion:** The v2 LBM+enthalpy stack with Gaussian2D does not yet match macrograph W/D on the benchmark grid. CI gates use `model_reference` (simulator self-consistency). Closing the macrograph gap requires finer grids (`high`/`ultra`), Goldak heat distribution, and/or additional calibration dimensions (heat loss, Marangoni scale).

## Open reference (FLOW-3D class)

FLOW-3D and similar FVM weld models report pool aspect ratios within ~15–25% of macrographs when fully calibrated. Our target for v2.0.0 is **±15–20% on `model_reference`**, not macrograph parity on `minimal` preset.

## Related tests

- `test_pool_geometry_standard` — ±20% vs `model_reference`
- `test_calibrated_pool` — ±30% vs `fit_metrics`
- `tools/fit_calibration` — grid search η, σ
- `tools/run_validation_matrix` — multi-speed matrix
