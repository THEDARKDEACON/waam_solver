# Calibration protocol

## What is fitted

Exactly one process-geometry card is global and reusable across meshes and
fidelity tiers:

* Arc-radius law `σ(I, V, CTWD)` — parameters `σ0`, `a_I`, `a_V`, `a_ctwd`
  (see `physics/heat_source.py::sigma_arc_m`).
* Tier C only: enhanced conductivity factor `λ` (`enhanced_k_factor`).

## What is NEVER fitted

Frozen from materials cards / literature (`tools/bayesian_calibrate.FIXED_PHYSICAL`):

* `dγ/dT`, `γ`, `C_darcy_phys`, `C_acc`, `evap_cooling_scale=1`, material `η`

Retuning these to pass a gate is forbidden (ground rule 1).

## Procedure

1. Fix production `dx` from a GCI study (`PASS_GCI`, clamps inactive).
2. Collect multi-observable data: fusion W/D, bead W/H, toe angle, pyrometer
   traces where available.
3. Fit the arc-radius card with `tools/bayesian_calibrate.py` (GP surrogate +
   MCMC with discrepancy terms).
4. Write `materials/calibration/<alloy>.global_card.yaml`.
5. Validate on **held-out** process points (`WAAM_HELDOUT_VALIDATION=1`).

## Equal-time / steady-state

Calibration runs use `measure.steady_state.run_to_steady_state` or equal
travel distance — never fixed `n_steps` (C3).

## Invalid historical cards

`materials/calibration/ER70S-6.auto_calibrate.yaml` and mesh-study JSON written
before the Phase-1 critical fixes are **superseded / invalid**.
