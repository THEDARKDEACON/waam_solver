# Spec: what sets penetration depth (stage P)

Status: proposed, 12 Nov 2026. Written before any stage-P run exists. Commit this file
unchanged before the first run; it is the pre-registration. Start only when no other
suite is running on the machine you use.

## Questions

1. Is the model's depth set by the arc share of the heat (η·P minus the droplet heat)
   and by the depth of the prescribed Goldak volume, and not by the flow?
2. Does the fused plate area agree with the paper where the peak depth does not, and
   is it better behaved on grid refinement than the depth?

This is a diagnostic. It changes no default, no job file and no fitted value. What to
do about the answer is the user's decision.

## Why

**Depth is not grid-converged** on run04 (server, n = 3): 0.725 / 0.600 / 0.530 mm at
dx 0.4 / 0.3 / 0.2 (`docs/MESH_REPORT_2026-11-12.md`).

**Depth responds to heat input about three times too strongly.** On the 14 held-out
beads at dx 0.4 (`docs/LAMBIASE_PREREG.md`, outcome table) the model's depth spans
0.60–2.20 mm and the measured penetration 0.88–1.72 mm. Against line energy the
log–log slope is 1.2 for the model and 0.3–0.4 for the measurement.

**Code reading gives a candidate cause** (not yet tested):

* With `arc_physics.heat_anchor: torch` the Goldak volume starts at the plate face and
  extends downward only (`solvers/coupled_step.py` `_torch_anchor_k`;
  `kernels/thermal.py` `_goldak_dk`, cells with dk > 0 are skipped). On a first-layer
  bead, all arc heat is therefore placed inside the plate, to a depth set by
  `c_mm` = 2.6. The bead above the face is heated by the droplets only.
* With `energy_split: true` the arc gets what is left after the droplets
  (`physics/arc.py` `_arc_power_W`). Wire feed is the same for all 24 beads, so the
  droplet heat is the same for all of them. Reviewer's hand estimate (droplets at
  liquidus + 400 K, stick-out heating ignored): droplets ≈ 1.49 kW; arc share 0.88 /
  1.21 / 1.38 kW at 2.9 / 3.3 / 3.5 kW. Power rises 21 %, the arc share 56 %.
* On the 14 held-out beads the model's depth follows that arc share per unit length
  (r = 0.94); the measured penetration does not (r = 0.39).

**The measured side** (corrects an earlier reviewer statement):

* The paper's Eq. 2 defines dilution as p/(h + p), a depth ratio. The `pool_depth_mm`
  values in the job files are therefore the paper's own penetration depth, to about
  ±0.04 mm from rounding. Model `D_mm` against it is a fair comparison.
* The paper's cross-section area exceeds the wire supply A_wire = π/4·d²·S_w/v by
  17–56 % on all 24 rows, while width × height × 0.70–0.93 equals the wire supply. The
  area column therefore appears to include the fused plate. On that reading
  A − A_wire is a measured fused plate area: 3.1–10.1 mm² on the 18 beads in use,
  5.7 mm² for run04, uncertain by about ±1 mm². It has not been used so far.
* **Assumption to confirm:** that reading rests on the table, not on the paper's
  Fig. 2, which this spec's author could not view. Stage P3 is provisional until the
  user confirms it from the figure.

## 1. Code changes (measurement and record only)

No solver, kernel, default or job-file change.

1. `measure/steady_section.py`: add `A_fused_mm2` and `A_fused_sd_mm2`. Per slice,
   count plate cells (`plate_k0` ≤ k < `nz_solid`) that are metal and have
   `T_max` ≥ liquidus, times dx²; mean and SD over the window's slices. Add
   `fullness` = `A_fused_mm2` / (`W_mm` · `D_mm`).
2. `tools/steady_bead.py`: add to each record
   * `Q_total_W`, `Q_droplet_W`, `Q_arc_W`: run means of what was actually injected
     (the value `_arc_power_W` returns, and the droplet power it subtracts), and
     `T_drop_K`;
   * `heat_k_minus_face`: run mean of `twin._last_heat_k` − (`nz_solid` − 1).
     Expected 0, meaning the whole volume lies below the plate face.

## 2. Checks before any GPU run

1. Solver untouched: single-thread CPU (`WAAM_CPU_THREADS=1`), `--hash-fields`,
   identical hash and identical existing record fields before and after the change.
2. Unit test for `A_fused_mm2`: a synthetic `T_max` with a semi-elliptical fused zone
   of known area is recovered within 5 % at dx 0.5, 0.4, 0.3 and 0.2.
3. Core CPU suite passes.

If check 1 fails, stop and report.

## 3. Stages

Common settings, as mesh stage M1:

```bash
J=jobs/examples/lambiase2026
C="--x0 24 --x1 28 --eta 0.819 --b 2.42 --u-design 2.0 --tau-min 0.5002"
```

Run time per bead is 44 mm / v: `--time-s 5.9` at 450, `4.4` at 600 and `3.5` at
750 mm/min (as `jobs/calibration/sweep_lambiase.yaml`). A run counts if it has no NaN
and `mach_frac_window` ≤ 2 %. Excluded runs are reported.

**P0 — heat partition, no GPU.** For the 18 beads (calibration 2, 4, 10, 12 and the
14 held-out), compute `energy_split_W` exactly and regress, on the existing dx 0.4
predictions, model `D_mm` and measured p against Q_arc/v and against η·P/v.

**P1 — depth against the prescribed volume** (laptop, 6 runs, about 1.2 h). dx 0.4,
run04 and run12, two repeats each:

```bash
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.4 --time-s 3.5 $C --c 1.3 --repeat 2 --out runs/steady/depth_P1.jsonl
python3 -m waam_twin.tools.steady_bead --job $J/run12.yaml --dx 0.4 --time-s 3.5 $C --c 1.3 --repeat 2 --out runs/steady/depth_P1.jsonl
python3 -m waam_twin.tools.steady_bead --job $J/run12.yaml --dx 0.4 --time-s 3.5 $C --repeat 2 --out runs/steady/depth_P1.jsonl
```

The run04 baseline (c 2.6) is the dx 0.4 set of P2. Check the unit of `--c` in the
tool's help before running.

**P2 — fused area on the grid sequence** (server, 9 runs, about 3 h). run04 at
dx 0.4, 0.3 and 0.2, three repeats each, into `runs/steady/server/depth_P2.jsonl`.
Server rules as in `docs/SERVER_MESH_PREREG.md`.

**P3 — the 18 beads at dx 0.4** (server, about 1.5 h; not the laptop, because run04 comes from P2 on the server and platforms are not mixed within one comparison). One run
per bead into `runs/steady/depth_P3.jsonl`; run04 is taken from P2. This uses the
mesh-study speed ceiling, so it is not a repeat of the pre-registered held-out
evaluation and does not replace it.

**Additions agreed before any stage-P run (12 Nov 2026).**

*Context:* the conduction-only grid check (`docs/CONDUCTION_DEPTH_PREREG.md`, run
before this spec was committed) gave depth 2.11 / 2.19 / 2.19 mm at dx 0.5 / 0.4 / 0.3
with no flow; the full model loses 17 % from 0.4 to 0.3. The heat source and depth
measurement are therefore not the cause of the depth drift.

**PT — conduction only at matched arc power** (laptop, 2 runs, about 1 h). Job
`runs/steady/diag/run04_conduction.yaml` (flow, free surface and wire off), dx 0.4 and
0.2, one run each (deterministic), same `$C` and `--time-s 3.5`, with
`--eta η_T`, where η_T = Q_arc(run04, exact from P0) / P_mean(run04). The value is
fixed by that formula from P0 before PT runs; it is not tuned. Output
`runs/steady/depth_PT.jsonl`.

**P1b — P1 on the near-converged grid** (server, 2 runs, alongside P2). run04,
dx 0.2, `--c 1.3`, two repeats, into `runs/steady/server/depth_P1b.jsonl`; the c 2.6
baseline is P2's dx 0.2 set.

*Notes:* P0's correlation was already seen by the reviewer (r = 0.94), so P0 confirms
that hand estimate with exact numbers and is not a blind test. A − A_wire assumes all
wire is deposited; at 95–98 % deposition efficiency it underestimates the measured
fused area by about 0.4–0.8 mm² for run04 (inside the stated ±1 mm²). For reference,
a semi-elliptical fused zone has `fullness` π/4 ≈ 0.785.

## 4. Criteria (fixed now)

* **P0.** "Depth follows the arc share" if r(model D, Q_arc/v) ≥ 0.85 and the model's
  log–log slope against η·P/v is at least twice the measured one. Otherwise not shown.
* **P1.** Elasticity E = ln(D at c 2.6 / D at c 1.3) / ln 2, per bead.
  E ≥ 0.5 on both beads → depth is prescribed by the heat volume.
  E ≤ 0.2 on both → depth does not depend on the volume's depth.
  Anything else → mixed, reported as such.
* **P2.** Change in `A_fused_mm2` from dx 0.3 to 0.2: converged if ≤ 6 % or within
  2·SE (the S3 rule). Also report the relative repeat SD of `A_fused_mm2` beside
  that of `D_mm` on each grid.
* **P3.** Per bead, r_A = `A_fused_mm2` / (A − A_wire) and r_D = `D_mm` / p; and the
  log–log slopes of both, model and measured, against η·P/v.
  * Median r_A within 0.75–1.33 and area slopes within a factor 1.5 → the plate gets
    about the right heat and the error is in where it is put. Depth too sensitive and
    width not sensitive enough to power are then one problem.
  * Median r_A < 0.75 → too little heat reaches the plate.
  * Median r_A > 1.33 → too much.
  * If the user finds the area column is reinforcement only, P3 reports the model
    numbers and makes no comparison.

* **PT.** Depth change dx 0.4 → 0.2 with conduction only at matched arc power:
  |Δ| ≤ 5 % → the depth drift in the full model is a flow effect. Also report
  D_PT/D_full per grid (full-model D from P2): ratio near 1 at dx 0.4 and falling at
  dx 0.2 → the flow takes depth away on refinement.
  (D_full is from the server and D_PT from the laptop; the ratio is descriptive, not a
  criterion, because platforms are not mixed in a test.)
* **P1b.** E at dx 0.2, read with the P1 bands; reported beside E at dx 0.4.

Reviewer's predictions, recorded so the result can contradict them: P0 holds;
P1 gives E ≥ 0.5 on both beads; P2 does not converge (area falls 10–20 % from dx 0.3
to 0.2) but its relative repeat SD is under half of depth's; P3 gives a model area
slope at least twice the measured one, with r_A below 0.75 at 2.9 kW.

## 5. Do not

* Change `heat_anchor`, `c_mm`, `energy_split`, the droplet superheat or any default;
  recalibrate; or re-score the pre-registered held-out checks.
* Try other values of `--c`, or add stages after seeing a result.
* Touch the user's uncommitted edit in `jobs/examples/bead_on_plate.yaml`.
* Edit kernel files while any suite is running.

## 6. Report back

Append an `## Outcome` section to this file, whatever the result:

1. The three checks of section 2, pass or fail.
2. P0: the exact heat partition per power level, the two correlations and the slopes.
3. P1: D, W, `A_fused_mm2` per bead and c (mean, n), E, and the reading; P1b and PT
   likewise.
4. P2: table of `A_fused_mm2`, `D_mm`, `fullness` (mean ± SD, n) per grid; verdict.
5. P3: per-bead table (P, v, p, D, A − A_wire, `A_fused_mm2`, r_D, r_A), medians,
   slopes, and the reading. Mark it provisional unless the user has confirmed Fig. 2.
6. Each prediction: held or not. Excluded runs and any departure from this spec.
7. A draft, not applied, of a note for the Limits section of
   `docs/VALIDATION_WRITEUP.md` stating what the result shows about depth. The
   recorded "penetration MAE 22.8 %, pass" stays as scored; the note sits beside it.

Then stop and wait for the user's decision.
