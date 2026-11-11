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

## Outcome (11 Nov 2026, 03:00) — runs/calibration/sweep_park_measured.*
1. **Physical consistency: PASS.** η posterior median 0.87, 90 % [0.77, 0.94];
   best fit 0.89 (was η_eff 1.00 with the assumed 2.73 kW input).
2. **Calibration fit: PASS.** Flat W 5.92 (−3 %), D 1.71 (−5 %), H 2.07 (−6 %);
   surrogate leave-one-out W/D error 0.24/0.14 mm (was 0.46/0.21).
3. **Held-out re-tests** (best fit; band = 4 posterior draws):

| bead | W | D | H |
|---|---|---|---|
| overhead | 4.90 vs 5.7 (−14 %) ✓ | 1.98 vs 1.9 (+4 %) ✓ | 1.18 vs 2.5 (−53 %), band [1.18, 3.36] — unsteady, not resolved |
| vertical down | 6.13 vs 6.9 (−11 %) ✓ | 1.86 vs 0.8 (**+132 %**) ✗ | 1.95 vs 1.8 (+8 %) ✓ |
| Bruno | 5.65 vs 4.33 (**+31 %**) ✗ | — | 2.34 vs 2.14 (+9 %) ✓ |

6 of 8 measured held-out outputs pass (4 of 8 before); the two failures are
the ones predicted in advance — vertical-down depth (torch-fixed heat volume
cannot show the pool running ahead) and Bruno width (Goldak b from Park's
bead does not transfer to another process). Overhead height remains unsteady
(frozen height varies 1.2–3.4 mm between draws).
