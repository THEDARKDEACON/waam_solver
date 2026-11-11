# Independent re-test on our own beads — protocol

The Park and Lambiase held-out beads have now been seen, so any model change after
those tests (mesh-study closures, recalibration) needs beads the model has never met.
This is that test. Fill in the data, then pre-register before running anything.

## Experiment (minimum useful set)
* Same wire, plate and gas for every bead; plate ≥ 6 mm thick, ≥ 70 × 30 mm, clean.
* **3 powers × 2 travel speeds × 2 repeats = 12 beads**, straight, ≥ 60 mm long.
  Choose powers that keep one transfer mode, or record the mode for each bead.
* Log U(t) and I(t) at ≥ 1 kHz during each bead (the welder's display is not
  enough); record wire feed, travel speed and CTWD as set.
* If possible, film a few droplets at ≥ 2000 fps to measure the impact speed
  (job key `process.droplet_impact_speed_m_s`; the twin's default is uncited).
* Section each bead at mid-length (steady part), polish, etch (2 % nital), photograph
  with a scale bar. Measure width, height, penetration and, if possible, reinforcement
  area (ImageJ). One section per bead; repeats give the scatter.

## Data entry
One row per bead in `docs/validation/data/own_beads/own_beads.csv` (repeats share a
`bead_id`); the columns are named in its header. `mean_power_W` is the mean of U(t)·I(t),
not mean U × mean I. Then:

```bash
python -m waam_twin.tools.make_own_bead_jobs     # → jobs/examples/own_beads/<bead_id>.yaml
```

Each job carries the repeat mean as the reference and the repeat SD as `*_sd`.

## Test (pre-register before any run)
Write `docs/OWN_BEADS_PREREG.md` and commit it before the first simulation:
* **Model frozen**: the model and η/b from the last calibration, no refit. This is
  a pure prediction, the strongest test available.
* Criteria (as Lambiase): MAE ≤ 10 % height, ≤ 15 % width, ≤ 25 % penetration;
  signs of dH/dv, dW/dv, dW/dP; report each bead against its repeat scatter.
* Run: `tools/steady_bead.py` per bead at the calibrated grid, 2 repeats each.
* Report the result whatever it is.
