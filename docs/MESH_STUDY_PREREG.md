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
