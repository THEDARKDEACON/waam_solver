# Park re-calibration with measured electrical input — pre-registration
(written 11 Nov 2026, before any run of the jobs below)

## What changed and why
Park et al. 2019 (Appl. Sci. 9:4626, local PDF) plot the measured current and
voltage for each position (Fig. 3b flat, 4b overhead, 8b vertical down).
Digitised over 7 pulse periods each (docs/validation/data/park2019_waveforms/,
overlays checked by eye; pulse frequency recovered within 1-2 % of 151.5 Hz):

| position | I_mean A | I_rms A | U_mean V | mean(U·I) W |
|---|---|---|---|---|
| flat | 138.6 | 178.7 | 20.36 | 3405 ± 109 |
| overhead | 140.5 | 180.7 | 20.38 | 3446 ± 98 |
| vertical down | 138.2 | 178.4 | 19.44 | 3260 ± 84 |

The earlier jobs used a rectangular 345/35 A train (I_mean 91 A, I_rms 150 A)
and a guessed 22 V → 2.73 kW. The real pulse has ramps and a ~95 A shoulder;
91 A could not melt 7 m/min of Ø1.0 mm wire. The earlier fit's η_eff ≈ 1.0
(η·P ≈ 2.73 kW) corresponds to η ≈ 0.80 of the measured power — inside the
calorimetric window. That consistency is a prediction this run tests.

New job keys `process.mean_power_W` and `process.current_rms_A` (job.py);
new jobs `bead_park_measured{,_overhead,_vertical_down}.yaml` (each with its
own measured waveform); sweep `jobs/calibration/sweep_park_measured.yaml`.

## Fixed in advance
* Model, mesh, Goldak shape, window: as `sweep_park_goldak.yaml`.
* One fitted value, η, prior [0.70, 0.95], flat bead only, n = 6, seed 11.
* Overhead run to 3.5 s (window freezing), vertical down 2.5 s, Bruno 4.0 s.
* Bruno inputs unchanged (measured 110 A / 19.7 V steady); it receives the
  calibrated η.

## Criteria
1. **Physical consistency** — η posterior median inside 0.75–0.90.
2. **Calibration fit** — flat W, D, H within 10 % at the best fit.
3. **Held-outs** — each output within 15 % (or within the 90 % prediction band).
   Overhead, vertical down and Bruno were seen in the earlier validation
   (docs/PARK_VALIDATION.md), so passes here are a *re-test after a
   measured-input correction*, not an independent validation. The earlier
   diagnosis predicts vertical-down depth will still fail (structural).

No parameter is changed after seeing these results; any further change needs
beads not yet used.
