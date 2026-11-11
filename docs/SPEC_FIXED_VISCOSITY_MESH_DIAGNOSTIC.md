# Spec: fixed-viscosity mesh diagnostic (stage V10)

Status: proposed, 11 Nov 2026. Written before any V10 run exists, while mesh stage M3
was still running. Start only after M3 has been closed out under
`docs/MESH_STUDY_PREREG.md`. Commit this file unchanged before the first V10 run; it
is the pre-registration.

## Question

Does penetration depth stop changing with the grid when the liquid viscosity is a
fixed multiple of the molecular value and Smagorinsky is off?

This is a diagnostic. It does not adopt a new closure, change a default, or
recalibrate anything. Whether to adopt is the user's decision, taken after the result.

## Why

Mesh stages M1 and M2 (run04, means over repeats, `runs/steady/mesh_M{1,2}.jsonl` in
the run worktree):

| | D at dx 0.5 / 0.4 / 0.3 | change 0.4 → 0.3 | W change | τ |
|---|---|---|---|---|
| M1, C_s 0.17 | 0.672 / 0.747 / 0.607 | −18.7 % | −4.0 % | 0.50022 / 0.50028 / 0.50037 |
| M2, C_s 0 | 0.632 / 0.718 / 0.344 | −52.1 % | −6.0 % | same |

Caveat on the M1 row: its dx 0.3 depth is the single run that passed the 2 % window
rule. Counting both repeats (0.61 and 0.74 mm), M1's depth change is −9.7 %, which
already lies inside the 10 % band of section 4. A V10 "converged" alone therefore
cannot show that the fixed viscosity did anything; the control arm C (section 3)
supplies the comparison under the same repeats and rule.

Depth holds only while the eddy viscosity damps the flow, and that viscosity shrinks
with the cell (ν_t/ν 0.47 → 0.34 → 0.23 over dx 0.5 → 0.3), so each grid solves a
different problem. A viscosity fixed in physical units removes that link. At ×10 the
cell Reynolds number at dx 0.4 falls from a few hundred to a few tens.

No such run is on record. Every steady and mesh record has τ − 0.5 between 0.0002 and
0.0009 (molecular viscosity). The 22 Oct Smagorinsky sweep (`flow_diag.jsonl`,
handoff §9) was one grid, before the forcing fix. The "viscosity floor" runs
(handoff §16) changed the time step, not the viscosity.

Caution on record: `reports/WAAM melt pool model calibration.md` advises against
calibrating an enhanced viscosity in a flow-resolving model (Kidess et al.: the factor
can stand in for numerical diffusion). That is why this stays a diagnostic.

## 1. Code change

Add one multiplier on the liquid dynamic viscosity, default 1.0:

* job key `advanced_physics.viscosity_scale`, in `jobs/schema.json`;
* `tools/steady_bead.py --visc-scale`, recorded in each record as `viscosity_scale`.

Viscosity reaches the relaxation time in three places: `grid.py:153` (scalar τ and the
dt choice), `twin.py:498` (plate τ), and the μ(T) table read in
`kernels/thermal.py` `refresh_properties*`. Scaling the loaded material (scalar `mu`
and the μ(T) table) before the grid and GPU tables are built covers all three with no
kernel edit; use another route only if that one fails a check below.

Nothing else changes: conductivity, `C_darcy`, surface tension, Marangoni, the
τ floor and the speed ceiling all stay as in M1.

## 2. Checks before any GPU run

1. Scale 1.0 is bit-identical to the current code: single-thread CPU
   (`WAAM_CPU_THREADS=1`), `--hash-fields`, with and without the flag.
2. Core CPU suite passes (78 tests), including `trt_poiseuille` and `lbm_poiseuille`.
3. Smoke at dx 0.5, 0.3 s, `--visc-scale 10 --cs 0`: the `CEILING` line shows
   dt 20.00 µs (unchanged from M1) and τ ≈ 0.5022; no NaN.
4. The per-cell field follows: median `tau_field − 0.5` over liquid cells is about
   10× its value at scale 1.0.

If check 1 or 4 cannot be met, stop and report. Do not run the stage.

## 3. Runs

From the run worktree, at the commit that holds the code change:

```bash
J=jobs/examples/lambiase2026
C="--time-s 3.5 --x0 24 --x1 28 --eta 0.819 --b 2.42 --u-design 2.0 --tau-min 0.5002 --cs 0 --visc-scale 10"
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.5 $C --repeat 2 --out runs/steady/mesh_V10.jsonl
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.4 $C --repeat 3 --out runs/steady/mesh_V10.jsonl
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.3 $C --repeat 3 --out runs/steady/mesh_V10.jsonl
```

Control arm C (amendment before any run, 11 Nov): the M1 model under the same rule,
`--cs 0.17`, no `--visc-scale`, into `runs/steady/mesh_V10C.jsonl`. No new runs at
dx 0.5. At dx 0.4 and dx 0.3, run whatever is needed to bring the counted M1 runs up
to three per grid, reusing the counted M1 records:

```bash
CC="--time-s 3.5 --x0 24 --x1 28 --eta 0.819 --b 2.42 --u-design 2.0 --tau-min 0.5002"
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.4 $CC --repeat 1 --out runs/steady/mesh_V10C.jsonl
python3 -m waam_twin.tools.steady_bead --job $J/run04.yaml --dx 0.3 $CC --repeat 2 --out runs/steady/mesh_V10C.jsonl
```

About 1.4 h. The control is judged by the same section 4 rule (C has `--cs 0.17`;
V10 has `--cs 0`).

V10 alone takes about 2.6 h on the RTX 3050 (6, 12 and 36 min per run in M1). Three repeats on the
two fine grids because depth repeats at dx 0.3–0.4 have differed by up to 0.16 mm.

A run counts if it has no NaN and `mach_frac_window` ≤ 2 % (mesh amendment 2).
Excluded runs are reported. If a grid is left with fewer than two valid runs, add up
to two replacement runs at that grid, then stop.

## 4. Criteria (fixed now)

Primary, depth of run04. Δ = (D̄₀.₃ − D̄₀.₄) / D̄₀.₄, with
SE = √(s₀.₄²/n₀.₄ + s₀.₃²/n₀.₃) from the repeats.

* **Converged**: |Δ| ≤ 10 %.
* **Not converged**: |Δ| > 10 % and |D̄₀.₃ − D̄₀.₄| > 2·SE.
* **Inconclusive**: |Δ| > 10 % but within 2·SE. Add repeats at both grids, to at
  most five each, and apply the rule once more. Change nothing else.

**Attribution (fixed now)**, from the V10 and C verdicts together:

* V10 converged, C not converged → the grid dependence of depth comes from the
  resolution-dependent flow (eddy viscosity shrinking with dx). Caveat: ×10 also
  raises Pr tenfold and slows the Marangoni flow, so this shows the *damped* flow
  converges, not that the physical flow is resolved.
* V10 converged, C converged → with three repeats the M1 depth is already inside the
  band; nothing is attributed to viscosity, and the M1 "not converged" is reported as
  repeat noise at n = 1.
* V10 not converged (whatever C does) → depth's grid dependence is **not** a
  flow-resolution effect. The candidates are the heat-source discretisation, the
  free-surface curvature, or the contact line (M3). This is the more informative
  outcome and closes the viscosity route.
* Either arm inconclusive after five repeats → reported as inconclusive; no attribution.

The band is 10 %, not the 6 % of the mesh study, because 6 % of this depth is
0.04 mm, below what three repeats can resolve.

Secondary, reported with the mesh-study rule (|change 0.4 → 0.3| ≤ 6 %): W, W_bead,
H_frozen. Also report per grid: repeat SD of each output, `mach_frac_window`,
`u_pool_max_ms`, τ, area ratio, `booked_over_fed`, `gain_over_booked`.

Reviewer's predictions, recorded so the result can contradict them: depth converges;
width still narrows by 4–6 % per step (a toe effect, not bulk flow); repeat scatter
in H_frozen at dx 0.4 falls below M1's 0.19 mm; speed-cap hits vanish.

## 5. Stage 2, conditional

Only if the primary verdict is **Converged**: repeat section 3 on `run12.yaml` into
the same file (another 2.6 h), same criteria, and report the W_bead difference
run12 − run04 per grid (measured +0.5 mm). If **Not converged**, stop: a fixed
viscosity is rejected as the depth fix.

## 6. Do not

* Try other multipliers, or tune ×10 after seeing a result. A blow-up at ×10 is a
  result; report it.
* Raise conductivity, change `C_darcy`, or turn Smagorinsky back on in this stage.
* Change any job file, default or lock; recalibrate; or re-run held-outs.
* Touch the user's uncommitted edit in `jobs/examples/bead_on_plate.yaml`.
* Start the related ideas here: curvature cap in physical units
  (`FS_KAPPA_MAX`, `solvers/coupled_step.py:22`), a clamp-location diagnostic, and
  split sub-model convergence tests are separate jobs.

## 7. Report back

Append an `## Outcome` section to this file, whatever the result:

1. The four checks of section 2, pass or fail.
2. Table of D, W, W_bead, H_frozen (mean ± SD, n) per grid for V10 and C, beside the
   M1 and M2 rows; the attribution reading from section 4.
3. The primary verdict, with Δ and SE, and each secondary output against 6 %.
4. Each of the four predictions: held or not.
5. Absolute error against Table 4 for run04 (H 3.8, width 3.9, contact 6.0, p 1.07 mm)
   at dx 0.4, next to M1's, so the cost of the extra viscosity in realism is visible.
6. Excluded runs and why; anything that departed from this spec.

Then stop and wait for the user's decision on adoption.

## Outcome (12 Nov 2026, 02:24)
Records `runs/steady/mesh_V10.jsonl` (×10, C_s 0; laptop, worktree a0dbf03) and
`runs/steady/mesh_V10C.jsonl` (control; plus the counted M1 records at dx 0.4/0.3).

**1. Checks.** (1) scale 1.0 hash-identical to no flag: pass. (2) 78/78 core CPU tests:
pass. (3) dx 0.5 smoke: dt 20.00 µs, τ 0.5022, no NaN, no cap hits (GPU, 0.3 s): pass.
(4) per-cell τ − 0.5 median ratio 10.000: pass.

**2. Results, run04** (mean ± SD, n):

| | dx 0.5 | dx 0.4 | dx 0.3 | Δ 0.4→0.3 |
|---|---|---|---|---|
| V10 D | 0.626 ± 0.060 (2) | 0.593 ± 0.112 (5) | 0.523 ± 0.073 (5) | −11.7 % |
| V10 W | 4.63 ± 0.09 | 4.75 ± 0.03 | 4.59 ± 0.21 | −3.3 % |
| V10 W_bead | 4.72 ± 0.00 | 4.87 ± 0.10 | 4.67 ± 0.25 | −4.2 % |
| V10 H_frozen | 2.78 ± 0.11 | 2.84 ± 0.08 | 2.97 ± 0.05 | +4.7 % |
| C D | — | 0.771 ± 0.042 (3) | 0.651 ± 0.046 (3) | −15.6 % |
| C W | — | 4.67 ± 0.10 | 4.43 ± 0.03 | −5.1 % |
| C W_bead | — | 4.79 ± 0.08 | 4.51 ± 0.03 | −5.7 % |
| C H_frozen | — | 3.09 ± 0.17 | 3.14 ± 0.12 | +1.5 % |
| M1 D (all runs) | 0.672 | 0.747 | 0.674 | −9.7 % |
| M2 D | 0.632 | 0.718 | 0.344 | −52.1 % |

**3. Verdicts.**
* V10 primary (depth): at n = 3, Δ −10.2 %, |diff| 0.058 < 2·SE 0.164 → inconclusive;
  2 repeats added at each grid (n = 5), rule applied once more: Δ −11.7 %,
  |diff| 0.069 < 2·SE 0.119 → **inconclusive** (final).
* Control C: Δ −15.6 %, |diff| 0.120 > 2·SE 0.073 → **not converged**.
* Attribution (section 4): V10 inconclusive after five repeats → **no attribution**.
* Secondary, V10 (6 % band): W −3.3 %, W_bead −4.2 %, H_frozen +4.7 % — all within.
* V10 depth repeats are **bimodal**: at dx 0.4, 0.71/0.72/0.71 vs 0.51/0.49/0.53; at
  dx 0.3, 0.60/0.60 vs 0.46/0.48/0.48. The two clusters differ by ~0.2 mm on both
  grids; the grid-to-grid shift within each cluster is ~0.1 mm (−15 %/−9 %). The run-to-
  run state, not the grid, dominates depth scatter under ×10.

**4. Predictions.** Depth converges: not shown (inconclusive). Width still narrows
4–6 %: partly held (W −3.3 %, W_bead −4.2 %; the n = 3 value −1.5 % was not robust).
H_frozen scatter at dx 0.4 below 0.19 mm: held (0.08). Speed-cap hits vanish: held
(window ≤ 0.03 %, max 6 steps per run).

**5. Realism cost at dx 0.4** (Table 4, run04: H 3.8, width 3.9, p 1.07 mm):
H_frozen 2.84 (−25 %) vs M1 3.03 (−20 %); W_bead 4.87 (+25 %) vs M1 4.76 (+22 %);
D 0.59 (−45 %) vs M1 0.75 (−30 %). ×10 makes every dimension worse.

**6. Departures.** None from sections 3–4; the V10 runs and the control ran on the
laptop. Stage 2 (run12) not run: the primary verdict was not "converged".

Conclusion: a fixed physical viscosity does not resolve the depth question; it
removes speed-cap episodes, roughly halves the width narrowing at n = 3 (not robust
at n = 5), costs realism, and exposes a bimodal depth state. Not adopted (and adoption
is the user's decision in any case).
