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
3. Fit the free heat-source parameters with `tools/calib_sweep.py` (below).
   `tools/bayesian_calibrate.py` holds the arc-radius card format, but its
   surrogate is still a placeholder formula, not simulations.
4. Write `materials/calibration/<alloy>.global_card.yaml`.
5. Validate on **held-out** process points (`WAAM_HELDOUT_VALIDATION=1`).

## Equal-time / steady-state

Calibration runs use `measure.steady_state.run_to_steady_state` or equal
travel distance — never fixed `n_steps` (C3).

## Invalid historical cards

`materials/calibration/ER70S-6.auto_calibrate.yaml` and mesh-study JSON written
before the Phase-1 critical fixes are **superseded / invalid**.

## Sweep tool (`tools/calib_sweep.py`)

For calibration with thin data: a Latin-hypercube sweep of steady beads
(`tools/steady_bead`), one Gaussian-process surrogate per output and process
point, then the range of parameters consistent with the measurements rather
than one best fit. Spec files live in `jobs/calibration/` (example:
`sweep_park.yaml`).

```bash
export PYTHONPATH=$PWD WAAM_BACKEND=cuda      # from the parent of waam_twin/
python -m waam_twin.tools.calib_sweep plan     waam_twin/jobs/calibration/sweep_park.yaml
python -m waam_twin.tools.calib_sweep run      waam_twin/jobs/calibration/sweep_park.yaml   # resumable
python -m waam_twin.tools.calib_sweep fit      waam_twin/jobs/calibration/sweep_park.yaml
python -m waam_twin.tools.calib_sweep validate waam_twin/jobs/calibration/sweep_park.yaml
```

`fit` reports, in this order:

* **Surrogate leave-one-out error**: must be well below the measurement σ,
  or add samples.
* **Sobol indices** (total / first-order) per output: a parameter with total
  index < ~0.05 does not affect what you measured; fix it from physics.
* **Posterior**: median and 90 % interval per parameter, with measurement σ,
  surrogate σ and a model-discrepancy allowance combined. "NOT constrained"
  (interval ≥ half the prior width) means the data cannot determine it.
* Runs where clamps fired on > 1 % of steps are counted: they measure the
  numerical speed cap, not physics (run the mesh study first).

`validate` runs the best fit plus posterior draws at the `heldout:` points and
reports the prediction band and error where a measurement exists. Points
without a measurement are predictions only, and they turn into real tests
once those beads are measured.
