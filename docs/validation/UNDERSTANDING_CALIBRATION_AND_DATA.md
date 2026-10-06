# Understanding calibration data: macrographs, multipass, and trust

This note explains **what experimental data the twin uses**, **which pieces are for calibration vs validation**, and **how confident you should be for real-world use**. It is written for operators and students, not only for developers.

Related detail docs:

- Locked ER70S-6 case: [reference_case_ER70S6.md](reference_case_ER70S6.md)
- Second macrograph slot: [heldout_macrograph_slot.md](heldout_macrograph_slot.md)
- Material status: [MATERIALS.md](../MATERIALS.md)

---

## 1. The big picture in one diagram

```text
  ┌─────────────────────┐         ┌──────────────────────────┐
  │  CALIBRATION data   │         │  VALIDATION data         │
  │  (tune the twin)    │         │  (test without retuning) │
  ├─────────────────────┤         ├──────────────────────────┤
  │ 1× bead macrograph  │         │ Held-out macrographs     │
  │  W≈7 mm, D≈3 mm     │         │ (different I or speed)   │
  │  → fit η, Goldak,   │         │ Multipass remelt / HAZ   │
  │    recoil, etc.     │         │ Thermocouples (optional) │
  └──────────┬──────────┘         └────────────┬─────────────┘
             │                                 │
             ▼                                 ▼
      “Does the model match          “If we change the process,
       THIS coupon?”                  does it still predict?”
```

**Short answer to “are these for calibration?”**

| Data | Role |
|------|------|
| **Primary bead macrograph** (one process window) | **Yes — calibration / locking** |
| **Held-out macrographs** (other speeds/currents) | **No — validation / prediction test** (do not retune to them) |
| **Multipass remelt / HAZ measurements** | **Mostly validation** of layer-to-layer physics (can also guide tuning later, but today they are comparison targets) |

---

## 2. What is a macrograph?

A **macrograph** is a polished cross-section of a weld bead, photographed or measured under a microscope/loupe.

From that cut you typically measure:

| Symbol | Meaning | How you see it |
|--------|---------|----------------|
| **W** (width) | Fusion zone width at the plate surface | Left–right extent of melted metal |
| **D** (depth) | Penetration into the substrate | How deep the melt went below the original surface |
| Sometimes **H** | Bead reinforcement height | Crown above the plate |

In the twin, those numbers appear in the job YAML as:

```yaml
reference:
  pool_width_mm: 7.0
  pool_depth_mm: 3.0
  source: macrograph_ER70S-6_bead_on_plate
```

The simulator reports its own pool **W** and **D** from the liquid/fusion zone. Gates compare:

```text
error ≈ |W_model − W_macro| / W_macro   (and same for D)
```

For the locked ER70S-6 case the model is about **6.8 × 3.2 mm** vs macro **7 × 3 mm** (~**6.7%** error). That is what “calibrated to the macrograph” means.

### Why one macrograph is not enough for “real world”

Fitting η, Goldak shape, and recoil so **one** coupon matches is like tuning a recipe until dinner tastes right **once**. You still need other coupons (held-outs) to check you did not overfit.

---

## 3. What is multipass data?

**Multipass** (or multi-layer) means depositing a **second bead on top of a first** (or beside it), as in WAAM builds.

Measurements that matter:

| Quantity | Meaning |
|----------|---------|
| **Remelt depth** | How far the second pass remelts into the previous bead / substrate |
| **HAZ width / height** | Region that got hot enough to change microstructure (often from peak-T maps or etch) |
| Interpass temperature | Plate temperature when the next pass starts |

In this repo the two-layer job is:

- `jobs/examples/bead_calibrate_twolayer.yaml`
- Report tool: `python3 -m waam_twin.tools.multipass_report`

The twin can **predict** fusion/HAZ extents from `T_max`. Absolute gates stay soft until you paste **measured** `remelt_depth_mm` / `haz_width_mm` into the job `reference` (see `awaiting_measurement` flags).

**Multipass is not the primary pool-shape calibration.** The bead W/D macrograph is. Multipass checks whether heat storage and remelting behave sensibly when layers stack — critical for WAAM, but a **different** experiment.

---

## 4. Calibration vs validation (held-out) — plain language

### Calibration (fitting / locking)

You **are allowed** to adjust knobs so the model matches the calibrate coupon:

Examples of fitted knobs (locked in `bead_calibrate.yaml`):

- Arc efficiency η  
- Goldak ellipsoid sizes  
- Recoil accommodation `C_acc`  
- Evaporative cooling scale  
- Process: 100 A × 15 V, 6.5 mm/s  

Material file: `materials/validated/ER70S-6.v1.yaml` (`status: calibrated` for **this** window).

### Validation / held-out (prediction)

You **must not** retune those knobs. Change only the process (e.g. travel 11 mm/s or 120 A), run the twin, and compare to a **new** macrograph.

| Job | What changes | Purpose |
|-----|--------------|---------|
| `bead_calibrate.yaml` | — | Calibration lock |
| `bead_calibrate_heldout_fast.yaml` | Faster travel | Prediction check |
| `bead_calibrate_heldout_hot.yaml` | Higher current | Prediction check |
| `bead_calibrate_heldout_macro2.yaml` | Slower travel (5 mm/s) | **Awaiting your measured W/D** |

Until macro2 (and similar) are filled with real cuts, absolute “we predict new welds” claims stay weak — the suite can only check **trends** (e.g. slower travel → larger pool).

---

## 4b. Mesh tiers (shop vs HPC) — do not mix locks

Fitted knobs (η, Goldak, recoil, …) are **mesh-tier locked**. Coarse and fine calibrations are different locks; copying knobs across tiers invalidates held-out prediction.

| Tier | Jobs | Mesh / collision | Claim notes |
|------|------|------------------|-------------|
| **shop** | `bead_calibrate.yaml` + `bead_calibrate_heldout_*.yaml` | `dx_mm: 0.4`, SRT + variable-τ, `auto_dt_ma: false` | Absolute W/D claimed only after fit on this mesh |
| **hpc** | `bead_calibrate_hpc.yaml` + `bead_calibrate_hpc_heldout_*.yaml` (alias: `bead_physics_accuracy.yaml`) | `dx_mm: 0.2`, two-rate MRT + variable-τ, `auto_dt_ma: true`, `u_mach_limit_lu: 0.20` (needed so τ≥0.505 and Ma can both be met) | Re-fit before claiming absolute W/D; until then `absolute_wd: false` |

Every calibrate / held-out job carries a top-level `claim:` block (`mesh_tier`, `pool_metric`, accuracy phrases, fitted vs predicted knobs, out-of-scope physics). `assert_physics_lock` also fingerprints `dx_mm`, `dt_scale`, collision path, Ma/force caps, and `pool_metric`.

**Primary pool metric:** `claim.pool_metric: fusion_zone` — liquidus envelope from `T_max` (macrograph-like). `f_l` bounding-box W/D is still logged as secondary.

**Characterisation (honest one-liner):** mesh-locked absolutes; thermal advection is minmod-limited (not 1st-order upwind); shop = SRT+variable-τ, HPC = MRT+variable-τ; CSF toe uses solid-aware stencils; suitable as an engineering-grade process trend predictor.

```bash
# Shop lock + held-outs
python3 -m waam_twin.tools.prediction_report --tier shop

# HPC lock + held-outs (after GPU re-fit)
python3 -m waam_twin.tools.auto_calibrate --job jobs/examples/bead_calibrate_hpc.yaml --write
python3 -m waam_twin.tools.prediction_report --tier hpc
```

New solver flags of interest: `simulation.auto_dt_ma` / `u_design_m_s` (Ma-aware dt with τ≥0.505), `use_srt: false` + `use_variable_tau: true` (variable-τ MRT on HPC), CSF solid-neighbour curvature in `kernels/vof.py`.

---

## 5. Ambient temperature and conductivity (for the runs)

These are **not** calibrated from the macrograph; they come from the job/material files:

| Quantity | Value used | Where |
|----------|------------|--------|
| Ambient | **293 K** (~20 °C) | `process.T_ambient_K` in the job |
| ER70S-6 conductivity \(k\) | ~**28–34 W/(m·K)** vs T | `materials/validated/ER70S-6.v1.yaml` table |

The macrograph mainly constrains **pool shape** (energy coupling + heat-source shape + key forces), not every thermophysical number.

---

## 6. How confident should you be for real-world use?

| Use case | Confidence | Why |
|----------|------------|-----|
| Trends in the locked ER70S-6 window (faster → narrower, etc.) | **Medium–high** | Physics + one solid W/D lock |
| Absolute W/D for a **new** current/speed with no new cut | **Low–medium** | Held-out macros incomplete |
| Multipass remelt / HAZ absolute numbers | **Low** until measured references are filled | Predictions exist; experiments pending |
| Shop schedule with no experiment | **Low** | Overfitting risk |
| Recalibrate to **your** coupons, then nearby conditions | **Medium → higher** | Correct workflow |

**Practical rule:**  
*Macrograph = teach the twin one weld. Held-outs + multipass = prove it learned welding, not that one photo.*

---

## 7. What you should collect next (priority)

1. **Second bead-on-plate macrograph** at a different travel or current → fill `bead_calibrate_heldout_macro2.yaml` (`heldout_macrograph_slot.md`).  
2. **External datasets (wired):** Bruno GMAW surface W/H + PIONEER M1 wall — see [PIONEER_BRUNO_DATASET.md](PIONEER_BRUNO_DATASET.md).  
3. **Two-layer coupon**: remelt depth + HAZ size → fill `bead_calibrate_twolayer.yaml` `reference`.  
4. Optional: thermocouple traces behind the arc for cooling-rate credibility.

Commands after you have numbers:

```bash
cd /path/to/repo   # folder with pyproject.toml / jobs/
export PYTHONPATH=.
python3 -m waam_twin.tools.prediction_report --tier shop --with-bruno
python3 -m waam_twin.tools.prediction_report --tier hpc
python3 -m waam_twin.tools.multipass_report --job jobs/examples/wall_pioneer_m1.yaml
```

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| **Macrograph** | Cross-section photo/measurement of fusion zone (W, D, …) |
| **Calibration / lock** | Tuning model knobs to match one (or few) experiments |
| **Held-out** | Experiment reserved for testing; knobs stay frozen |
| **Multipass** | Second (or later) layer; remelt + HAZ matter |
| **HAZ** | Heat-affected zone — hot but not necessarily melted |
| **η (arc efficiency)** | Fraction of electrical power that enters the workpiece as heat |
| **Goldak** | Double-ellipsoid volumetric heat-source shape for the arc |
| **`model_reference`** | What the **simulator** produced on a known mesh |
| **`reference`** | What the **experiment** (or literature) measured |
| **Mesh tier** | Shop (`dx≈0.4`) or HPC (`dx≈0.2`) lock; knobs not interchangeable |
| **`claim.pool_metric`** | `fusion_zone` (primary) or `fl_bbox` (legacy liquid bbox) |
| **`auto_dt_ma`** | Shrink/raise `dt` so design velocity stays under Ma cap and τ≥0.505 |
