# WAAM Twin v2 — Validation Report

**Generated:** manual / CI (`WAAM_FULL_VALIDATION=1`)  
**Version:** `2.0.0`  
**Scope:** Regression + process smoke on CPU (`minimal` preset) unless noted.

## Run metadata

| Field | Value |
|-------|-------|
| Backend | `cpu` (CI), `cuda`/`vulkan` optional (`WAAM_BACKEND_MATRIX=1`) |
| Preset | `minimal` (CI), `standard` grid for pool calibration |
| Material | `ER70S-6` (`materials/validated/ER70S-6.v1.yaml`, `status: calibrated`) |
| Calibration | `materials/calibration/ER70S-6.bead_on_plate.yaml` |
| Job | `jobs/examples/bead_on_plate.yaml` |

## Kernel benchmarks (Phase 0–2)

| Test | Criterion | Status |
|------|-----------|--------|
| `thermal_diffusion` | L2 < 12% | Pass |
| `mass_conservation` | ρ drift < 2% | Pass |
| `lbm_poiseuille` | Profile L2 < 15% | Pass |
| `lbm_cavity` | Interior circulation | Pass |
| `stefan_solidification` | Front slope < 15% | Pass |
| `vof_mass` | φ drift < 2% | Pass |
| `laplace_csf` | CSF force non-zero | Pass |
| `marangoni_cell` | Flow vs dγ/dT sign | Pass |
| `heated_cavity` | Centre T rises | Pass |
| `arc_pressure` | Fz after Marangoni | Pass |
| `soak_10k` | No NaN, mass conserved | Pass |

## Process benchmarks (Phase 1–3)

| Test | Criterion | Status |
|------|-----------|--------|
| `pool_geometry` | W/D vs scaled ref < 35% (`minimal`) | Pass |
| `pool_geometry_standard` | W/D vs `model_reference` < 20% | Pass |
| `calibrated_pool` | Reproduce `fit_metrics` < 30% | Pass |
| `rosenthal_farfield` | Tail T < 55% | Pass |
| `thermocouple` | Probe T < 40% | Pass |
| `job_parity` | minimal vs standard grid < 35% | Pass |
| `multi_bead_width` | W vs `model_reference` < 15% | Pass |
| `two_layer_haz_ref` | T_max in [600, 3200] K | Pass |
| `two_layer_remelt` | Second layer remelts bead | Pass |
| `interpass_haz` | T_max > 600 K | Pass |
| `parametric_monotonic` | P, dwell monotonic | Pass |
| `moving_window` | Window shifts, no NaN | Pass |
| `surface_vtk` | Non-empty φ/f_l mesh | Pass |

## Validation matrix (Phase 4)

Run: `PYTHONPATH=. python3 -m waam_twin.tools.run_validation_matrix`

| Case | Model error gate | Notes |
|------|------------------|-------|
| bead @ 3, 5, 8 mm/s | ±15% vs `model_reference` | 88×44×44 @ 0.3 mm |
| multi_bead, two_layer | Path smoke (no W/D gate) | `minimal` preset |

Macrograph 7.0×2.8 mm is **not** a pass/fail gate — documented in [reference_case_ER70S6.md](reference_case_ER70S6.md).

## Backend smoke

| Backend | Test | Status |
|---------|------|--------|
| CPU | `test_backend_smoke` + full `run_all` | Required |
| Vulkan | `WAAM_BACKEND_MATRIX=1` | Best-effort |
| CUDA | `WAAM_BACKEND_MATRIX=1` | Best-effort |

## Known limits

- Gaussian2D arc on coarse grids under-predicts macrograph pool width.
- `minimal` preset (dx≈0.5 mm) cannot resolve 7 mm pools; use `model_reference` for CI.
- Full 266³ `standard` domain is not run in CI (runtime).

## Reproduce

```bash
cd FYP22-01
WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
WAAM_FULL_VALIDATION=1 WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
WAAM_STANDARD_VALIDATION=1 WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
```
