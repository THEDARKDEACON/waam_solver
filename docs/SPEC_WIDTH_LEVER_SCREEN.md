# Spec: width-lever screen on a wider window

Status: proposed, 2026-10-22. Follows section 7 of
`docs/HANDOFF_FSLBM_CALIBRATION.md`.

## Why

Fusion width sits at 5.19 mm against Park's 6.1 mm (-15%) at dx 0.4 mm.
Width repeats to 0.10 mm between identical runs, and the Marangoni scale (0.7 to
2.0) did not move it outside that scatter. Closing the gap needs about +0.9 mm,
so a knob that really controls width will show in a single run.

Frozen height still scatters (1.60 to 2.40 mm). It tracks the deposited area,
which ranges 0.87 to 1.15 of the wire-feed value run to run: metal pools unevenly
along the track, and a 4 mm window does not average over it.

This round answers one question: does any remaining knob move width by at least
0.5 mm? If not, width -15% is recorded as a model bias and the current knobs are
locked.

## 1. Measurement window

- `--time-s 4.0 --x0 22 --x1 34`. At 4.0 s the torch reaches the path end at
  x = 55 mm (`jobs/paths/bead_calibrate.csv` runs 15 -> 55 mm), so the window is
  21-33 mm behind it. The tool's path-end guard allows exactly 4.0 s.
- Window start stays at 22 mm, past the startup ramp near the 15 mm path start.
- Each run must report `steady=True`. A run that is not steady is reported but
  not used.

## 2. Tool change: `--arc-sigma`

Add `--arc-sigma MM` to `tools/steady_bead.py`. After the twin is built and
before `reset()`, set `twin.arc_sigma_m = MM / 1000` and
`twin.sigma_cells = twin.arc_sigma_m / twin.grid.dx`, the same way
`job.py` applies `arc_physics.sigma_mm`. Record `arc_sigma_mm` in the JSON
knobs.

Be clear about what this knob does with the Goldak source. With explicit Goldak
axes, `sigma_cells` does not change the heat input (it is only a fallback in
`Goldak3D._resolved_axes`). It sets the deposition footprint
(`deposition_footprint_cells`, `max(droplet radius, sigma * footprint scale)`),
the Lorentz current spread and the tracer spread. Arc pressure is unchanged,
because the job sets `pressure_sigma_mm` separately. A width change from this
knob therefore comes from where metal is placed, which is a geometric prior,
not from heat. Report it that way.

## 3. Runs (about 50 min of GPU, sequential, dx 0.4 mm)

Repo root, usual environment, all appending to `runs/steady/width_screen.jsonl`:

| tag | flags | what changes |
|-----|-------|--------------|
| base x2 | `--repeat 2` | new baseline on the wider window |
| contact60 | `--contact 60` | wetting: contact angle 80 -> 60 deg |
| b40 | `--b 4.0` | Goldak half-width 2.8 -> 4.0 mm (heat spread) |
| sigma30 | `--arc-sigma 3.0` | deposition footprint and Lorentz spread 2.0 -> 3.0 mm |

Common flags: `--job jobs/examples/bead_calibrate.yaml --time-s 4.0 --x0 22
--x1 34 --out runs/steady/width_screen.jsonl`. About 100 000 steps per run, about
10 min each.

Record for every run: W, D, H_frozen, W_bead, area_ratio, steady, mach steps.

## 4. Decision rules

Compare each screen run against the mean of the two new baseline runs.

- **Window check.** If the two baseline runs differ by more than 0.3 mm in
  H_frozen or more than 0.10 in area_ratio, the wider window did not settle the
  height. Report it, and use W and D only for the screen.
- **Lever found.** A knob with |dW| >= 0.5 mm whose depth stays within 10% of
  the baseline depth. If more than one qualifies, take the one with the larger
  dW per change in D. If it is `sigma30`, flag that the gain is a deposition
  prior before going on.
  - Bracket it with two more single runs, chosen to land on W 6.1 by linear
    interpolation through the screen values, then repeat the best value once.
  - Lock it: write the value into `bead_calibrate.yaml` and the three shop
    held-outs (`_heldout_fast`, `_hot`, `_macro2`), move the knob from
    `knobs_locked` to `knobs_fitted` in `claim:`, and update
    `model_reference.note` with the fitted numbers and this window.
  - Rerun `test_job_locks`, `test_fslbm` and `test_physics_tier`, then the
    15 mm/s held-out on the same window protocol (`--time-s` limited by its
    path end, slices 30-36 mm). W, D and H must all fall relative to the
    calibrate job.
- **No lever.** If no knob reaches |dW| >= 0.5 mm, stop fitting. Keep eta 0.82,
  Goldak 2.0/3.8/2.8/2.6, marangoni 1.0, contact 80, sulphur 28. Update
  `model_reference.note` in `bead_calibrate.yaml` with the steady numbers from
  the new baseline and add `width_bias_minus_15pct_shop` to
  `claim.limitations`. Run `test_job_locks`.

## 5. Recording

Append a new section to `docs/HANDOFF_FSLBM_CALIBRATION.md` before "Known
risks / gotchas" with the run table, the decision and any job changes.

## Out of scope

- Sulphur and eta (left for later, after a lever is or is not found).
- The HPC tier and dx 0.3 re-runs.
- Changes to the physics.
