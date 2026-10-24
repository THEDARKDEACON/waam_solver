# Spec: frozen-bead height metric and repeat mesh gate

Status: proposed, 2026-10-22. Follows section 6 of
`docs/HANDOFF_FSLBM_CALIBRATION.md`.

## Why

The mesh check failed on height only. dx 0.3 mm gave H 2.86 mm against a
dx 0.4 mm mean of 2.24 mm (+28%). Width (+3%) and depth (-4%) held.

Height tracks how much of the measuring window has solidified:

| run | frozen fraction | H mm |
|-----|-----------------|------|
| dx 0.4, run 2 | 0.91 | 1.83 |
| dx 0.4, run 1 | 0.84 | 2.50 |
| dx 0.4, run 3 | 0.81 | 2.39 |
| dx 0.3 | 0.78 | 2.86 |

`steady_cross_section` takes the tallest phi column in each slice. With 9-22%
of the window still liquid, that column can be the pool tail, which bulges
above the frozen bead. Width and depth come from `T_max`, which is recorded
as the torch passes, so they do not depend on when the run stops. The working
hypothesis is that the height spread and the mesh shift are a measurement
timing artifact. This spec tests that hypothesis before any knob fit.

## 1. Metric changes in `measure/steady_section.py`

Keep every existing key so older JSON lines stay comparable. Add:

- **Column classification.** For each slice i and column j above the plate
  (k >= nz_solid):
  - metal: `flags != FLAG_GAS`, fill `phi` clipped to [0, 1].
  - A column is *frozen* when every metal cell above the plate has
    `flags == FLAG_SOLID`. A column with no metal is empty.
- **`H_frozen_mm`.** Per slice: the maximum fill sum over frozen columns,
  times dx. Slices with no frozen column with fill sum > 0 are excluded. Mean
  over valid slices. `n_slices_frozen` reports how many contributed.
- **`area_mm2`.** Per slice: the sum of phi over all metal cells above the
  plate, times dx squared. This is the deposited reinforcement area. Report
  the mean and `area_sd_mm2`. Liquid and solid metal both count, because area
  is set by wire feed, not by solidification.
- **`area_expected_mm2`.** Wire area times wire feed over travel speed:
  `pi/4 * d_wire^2 * v_wire / v_travel`. For bead_calibrate this is
  `0.785 * 116.7 / 10 = 9.16 mm2`. Report `area_ratio = area_mm2 /
  area_expected_mm2`. The function gets the expected area as an optional
  argument; the tool computes it from the job's `process` block.
- **`W_bead_mm`.** Per slice: the width of the frozen-or-liquid columns whose
  fill sum is >= 0.5 cell, times dx (toe-to-toe reinforcement width, not the
  fusion width). Mean over slices.
- **`H_mm` meaning is unchanged** (tallest column, any phase). Rename it in
  printed output to `H_max` so it is not mistaken for the new metric.
- **Steadiness gate.** `steady = frozen_frac >= 0.97 and n_slices_frozen ==
  n_slices`. Reported, not enforced, by the function.

Percentage errors against Park: add `err_Hf_pct` for `H_frozen_mm`. Keep
`err_H_pct` on `H_mm` for continuity. `err_max_pct` uses W, D and
`H_frozen_mm`.

## 2. Tool changes in `tools/steady_bead.py`

- Pass the expected area (from `process.wire_diameter_mm`,
  `process.wire_feed_m_min`, `process.travel_speed_mm_s`).
- Print `H_frozen`, `area`, `area_ratio`, `W_bead` and `steady` in the RESULT
  line. `_summarise` adds sigma for `H_frozen_mm` and `area_mm2`.
- If `steady` is false, print `WARNING not steady: frozen_frac=...` after the
  RESULT line. The run is still recorded.
- Reject `--time-s` values that would run the torch past the path end
  (path `jobs/paths/bead_calibrate.csv` runs x 15 -> 55 mm, so 4.0 s at
  10 mm/s). The driver already holds position, but a parked torch would heat
  the window.

## 3. Unit test `validation/test_steady_section.py`

Host-only (no GPU). Use a small fake twin whose `grid` fields expose
`to_numpy()` over numpy arrays (nx 12, ny 9, nz 10, dx 0.4 mm, nz_solid 4).
Measure a 3-slice window (i 4-6), so one liquid column moves `frozen_frac`
below 0.97:

1. Flat frozen bead, 3 cells high, 5 columns wide: `H_frozen_mm == 1.2`,
   `area_mm2 == 15 * 0.16`, `W_bead_mm == 2.0`, `steady` true.
2. Same bead with one liquid column of 5 cells in one slice: `H_mm` reads
   2.0 mm on that slice, `H_frozen_mm` stays 1.2, `area_mm2` rises by
   2 * 0.16 on that slice, `steady` false.
3. Partial fill (phi 0.5 in the top cell): `H_frozen_mm == 1.0`.

Add a `__main__` that runs the three cases, so it runs the same way as the
other validation scripts.

## 4. Repeat runs (about 40 min of GPU, sequential)

Run from the repo root with the usual environment. Window 22-26 mm, so it is
24-28 mm behind the torch at 3.5 s:

```
python -m waam_twin.tools.steady_bead --job jobs/examples/bead_calibrate.yaml --time-s 3.5 --x0 22 --x1 26 --repeat 2 --out runs/steady/spread_frozen.jsonl
python -m waam_twin.tools.steady_bead --job jobs/examples/bead_calibrate.yaml --time-s 3.5 --x0 22 --x1 26 --dx 0.3 --out runs/steady/mesh_dx030_frozen.jsonl
```

Expected cost: about 9 min per dx 0.4 run (87 500 steps), about 22 min for dx
0.3 (116 667 steps).

If any run reports `steady` false, rerun that case at `--time-s 4.0` before
judging.

## 5. Decision rules

Compare dx 0.3 against the dx 0.4 mean.

- **Mesh gate passes** when W, D and `H_frozen_mm` are each within 10%, and
  `area_ratio` is within 0.90-1.10 on both meshes. Then resume the width
  fit from the earlier plan (Stage A, `marangoni_scale` 0.7 / 1.5 / 2.0 plus
  a confirmation run), using `H_frozen_mm` and these runs as the baseline and
  `--time-s 3.5`.
- **Height still fails, area passes.** Height is genuinely mesh-dependent
  (crown shape, not deposited metal). Fit W and D only. Report
  `H_frozen_mm` and `W_bead_mm` but do not target them.
- **Area fails** (outside 0.90-1.10). Deposition is losing or creating metal,
  so no geometry fit is valid. Stop and investigate mass conservation first.

## 6. Recording

Append the results and the decision to `docs/HANDOFF_FSLBM_CALIBRATION.md`
as a new section after section 6. Do not edit jobs unless a fit is locked.

## Out of scope

- Changes to the physics.
- The long-running gate tests that read `model_reference.n_steps`.
- The HPC tier.
