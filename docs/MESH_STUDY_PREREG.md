# Area excess and mesh convergence — pre-registration
(written 11 Nov 2026, before any of the runs below)

## Why
All calibration and validation so far ran at dx 0.4 mm, so the fitted η and b
absorb whatever that grid does. An audit (11 Nov) found the bead getting taller and
narrower as dx shrinks, a deposited area up to 1.3× the wire supply at dx ≤ 0.25 mm,
and a speed cap firing on 9–38 % of steps by an amount that differs per grid. No
mesh study has been run since the surface-height race fix.

## Common settings
Beads: Lambiase run04 (2.9 kW) and run12 (3.5 kW), both 750 mm/min, SOD 8 — the
same speed at two powers, so the width–power slope (the Lambiase defect) is in the
study. Lambiase best fit η 0.819, Goldak b 2.42 mm, everything else as the jobs.
3.5 s, window x 24–28 mm. Every grid uses the same speed ceiling:
`--u-design 2.0 --tau-min 0.5002` (TRT floor lowered from 0.5005 so that 2 m/s
fits at dx 0.5; at 0.5005 the ceiling would be 0.8–1.4 m/s and differ per grid).
Records carry the area budget (`measure/area_budget.py`).
Runs execute from a separate worktree at the commit that adds this file.

## Stage A — area excess (run04, dx 0.25 mm, one run)
Diagnosis rules, applied to the budget fields:
* `gain_over_booked` > 1.05 → the free-surface scheme creates metal;
* `booked_over_fed` > 1.05 → the deposition step books too much;
* `area_above` > 1.05× expected but `area_net` within 5 % → plate metal lifted
  above the face (measurement, not mass);
* area profile varying > 10 % along the steady part → humping.
If none applies and the area ratio is within 1.05, the excess is not reproduced
under the uniform speed ceiling, and that is the result.

## Stage M1 — mesh study, closures as calibrated (Smagorinsky C_s 0.17)
dx ∈ {0.5, 0.4, 0.3} mm × {run04, run12} × 2 repeats = 12 runs.
* A run counts only if `mach_frac_window` = 0 (no speed-cap hit while the window
  is measured) and it has no NaN; others are reported and excluded.
* Per grid: mean of the repeats; noise = pooled repeat SD.
* Grid-convergence index by the Celik et al. (2008) procedure for non-uniform
  refinement ratios (r 1.25 and 1.33), safety factor 1.25, for W (fusion width),
  D, H_frozen, W_bead. Oscillatory or divergent sequences are reported as such.
* **Converged** (per output, per bead): |change 0.4 → 0.3| ≤ 6 % (the measurement
  scatter in the Lambiase table is 5–7 %), or within 2× the pooled repeat SD.
* Width–power slope (run12 − run04 in W_bead) reported per grid.

## Stage M2 — A/B: Smagorinsky off (C_s = 0)
Identical to M1 with `--cs 0`. Decision, fixed now:
* if every M2 run is stable (no NaN, no speed-cap hit in the window) **and** the
  0.4 → 0.3 changes are no larger than in M1 (or all within the converged band),
  C_s = 0 becomes the default for bead jobs;
* otherwise C_s stays at 0.17 and the instability is reported as a resolution limit.

## Stage M3 — contact line (pre-registered by amendment before it runs)
A grid-independent contact-line model (physical slip length in mm) is to be
implemented; its A/B test and criteria will be added here before any run.

## After the study
* Converged range found → recalibrate once (runs 2, 4, 10, 12, same spec) at the
  coarsest converged grid. The Lambiase held-outs have been seen, so re-running them
  is a seen-data consistency check, not a validation; independent validation needs
  new beads (the user's own).
* No converged range → keep the fixed-grid (dx 0.4 mm) model, report it as such with
  the GCI numbers.

Results are reported whatever they are.

## Amendment 1 — Stage M3 defined (11 Nov 2026, before any M3 run; M1/M2 running)
Contact-line model, commit 5506cec: `simulation.contact_slip_band_mm` gives the
contact-line slip region a fixed physical width ℓ instead of one cell. With ℓ = dx it
is bit-identical to the old rule (single-thread CPU hash check). ℓ is a
regularisation, not a measured slip length (real slip lengths are nanometres); fixing
it in mm is what removes the grid dependence of the spreading rate.

* M3 = M1 settings + `--slip-band 0.4` (ℓ = 0.4 mm, the width the calibrated
  dx 0.4 mm model already has), at dx 0.5 and 0.3 × {run04, run12} × 2 repeats.
  At dx 0.4 M3 is the same model as M1, so the M1 dx 0.4 records are reused.
* Adopt the band (ℓ = 0.4 mm default for bead jobs) if every M3 run is stable and
  the 0.4 → 0.3 changes in W, D, H_frozen, W_bead are no larger than in M1 (or all
  within the converged band); otherwise keep the one-cell rule and report.
* Combined with the M2 decision: the adopted closures form the model that is
  recalibrated; if both are adopted, a final confirmation set (both on, dx 0.4 and
  0.3, both beads, 2 repeats) is run before recalibration.

Also fixed in 5506cec (no effect on results): the droplet impact pressure returned
½mv²/area (N/m, ≈ 1e-3 Pa) and was applied on top of the droplet momentum already
carried by the deposited cells; it now returns ½ρv² [Pa] and applies only when that
momentum is off.

## Stage A result (11 Nov 2026, 10:23) — area excess NOT reproduced
run04, dx 0.25 mm, ceiling 2 m/s (`runs/steady/mesh_A.jsonl`, run worktree):
area 10.18 mm² vs 10.77 expected (ratio 0.95); deposition booked 0.996 of the
wire fed, domain gained 0.997 of what was booked (mass conserved). Along the bead
the steady part (x 22–50 mm) carries 10.0–11.1 mm² (mean ≈ expected); a start-up
hump of 12.4 mm² sits at x 14–18 mm and the window x 24–28 is on a slight dip.
None of the diagnosis rules applies: under a uniform speed ceiling there is no
fine-mesh area excess. (The audit's 1.3 came from runs before the race fix
and/or the start-up hump.) W 4.20, D 0.49, H_frozen 3.11 mm.

## Amendment 2 (10:24 — after stage A, before any M1 result exists)
Stage A hit the 2 m/s ceiling on 1.2 % of window steps although the sampled peak
speed was 0.75 m/s: isolated one-step spikes. The "zero hits" rule would likely
exclude every run, so it is replaced, before any M1 record exists, by:
**a run counts if `mach_frac_window` ≤ 2 %** (isolated spikes, against 9–38 % of
steps in the old studies); the fraction is reported for every run, and the analysis
is also shown with and without runs above 1 %.

## Stage M1 result (11 Nov 2026, 14:01) — not fully converged
Records `runs/steady/mesh_M1.jsonl`. 3 of 12 runs broke the 2 % window rule, with
7.7–12.6 % of window steps at the ceiling (the other nine are ≤ 0.9 %). These are
intermittent episodes, not isolated spikes, and two of them fall at dx 0.3, so each
bead has one counted run there.

Pre-registered analysis (runs with > 2 % excluded; the ≤ 1 % analysis is identical):

| bead | output | dx 0.5 | 0.4 | 0.3 | Δ 0.4→0.3 | verdict |
|---|---|---|---|---|---|---|
| run04 | W | 4.57 | 4.63 | 4.45 | −4.0 % | converged |
| run04 | D | 0.67 | 0.75 | 0.61 | −18.7 % | **not converged** |
| run04 | H_frozen | 2.96 | 3.03 | 3.20 | +5.5 % | converged |
| run04 | W_bead | 4.72 | 4.76 | 4.54 | −4.6 % | converged |
| run12 | W | 5.27 | 5.06 | 4.87 | −3.7 % | converged |
| run12 | D | 1.57 | 1.43 | 1.45 | +1.4 % | converged |
| run12 | H_frozen | 3.23 | 3.10 | 3.20 | +3.4 % | converged |
| run12 | W_bead | 5.39 | 5.24 | 4.93 | −5.9 % | converged |

Deposited area ratio stays within 1.0–1.1 at dx ≤ 0.4 for both beads (no excess). Most
sequences are divergent or oscillatory, so the Celik GCI values and the extrapolated
values are not meaningful; the convergence verdict rests on the pre-registered
Δ-band rule.

Sensitivity (all 12 runs counted): run04 D becomes −9.7 %, inside 2 × the repeat SD
(the two dx 0.3 repeats give D 0.61 and 0.74, so this depth scatter is the real noise at
that grid); run12 W_bead becomes −6.2 % and fails. So under either analysis exactly
one of the eight output–bead pairs fails, and in each case it fails at the edge of
the band.

**Systematic trend, within the band:** width falls on every refinement step for
run12 and from 0.4 to 0.3 for run04 (−4 to −6 %), and the width–power response
(run12 − run04 in W_bead) shrinks with refinement: +0.67 / +0.47 / +0.39 mm at
dx 0.5 / 0.4 / 0.3 (measured +0.5 mm). The fixed-dx calibration therefore does
not transfer to a finer grid without refitting; the closest match to the measured
width–power slope is at dx 0.4.

Verdict per the pre-registration: **M1 is not converged** (1 of 8 pairs fails in both
analyses), but only marginally. M2 and M3 are judged against these numbers.
