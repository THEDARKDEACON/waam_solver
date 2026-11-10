# Park calibration and held-out validation (10 Nov 2026)

Spec: `jobs/calibration/sweep_park_goldak.yaml`; records
`runs/calibration/sweep_park_goldak.{jsonl,log,report.json,draws.json}`.
Background: `docs/PARK_PHYSICS_CHECK.md`.

**Verdict: calibrated on the flat bead, NOT validated.** Two of three
held-outs fail outside their uncertainty, the third is not a valid comparison
(unsteady), and each failure points at a specific model limitation.

## Model

`jobs/examples/bead_park_goldak.yaml` (dx 0.4 mm, FSLBM, τ_odd = 1):

* honest energy budget — arc + droplets = η·mean(U·I);
  mean(U·I) = V·I_mean + R(I_rms² − I_mean²) = 2.73 kW (V 22 V estimated);
* pulsed forces on I_rms (150.5 A), one droplet per pulse (151.5 Hz, 4.4 mg);
* Goldak volume fixed to the torch (`heat_anchor: torch`), shape from the
  measured bead (b 3.0, c 2.6, a_f 2.0, a_r 3.8 mm);
* gravity as a direction (`welding_position`).

One fitted value: effective arc efficiency η_eff (Park's voltage is not
tabulated, so η·mean(U·I) is the uncertain input).

## Calibration (flat bead)

| | W | D | H frozen |
|---|---|---|---|
| Park measured | 6.1 ± 0.3 | 1.8 ± 0.15 | 2.2 ± 0.15 |
| best fit η_eff = 1.00 (surrogate) | 5.63 (−8 %) | 1.66 (−8 %) | 2.32 (+5 %) |

η_eff posterior median 1.02, 90 % [0.96, 1.12] — 2.6–3.1 kW into the
workpiece, at or above the top of the physical window (~1.9–2.7 kW for
η 0.75–0.88, V 20–26 V). Treat η_eff as part effective (it absorbs missing
transport physics), not as a measured efficiency.

Run-to-run scatter limits the fit: surrogate leave-one-out error (D 0.21 mm,
H 0.35 mm) is as large as the effect of η over the prior. Repeats per
sample, or longer averaging windows, are needed before a tighter fit means
anything.

## Held-out predictions (never used in the fit)

| Held-out | Output | Predicted (best; band) | Measured | Error |
|---|---|---|---|---|
| Park overhead (gravity +z) | W | 4.48 [4.27, 5.52] | 5.7 | −21 % |
| | D | 1.71 [1.60, 1.88] | 1.9 | −10 % ✓ |
| | H | 1.60 [0.38, 1.60] | 2.5 | −36 % |
| Park vertical-down (gravity ∥ travel) | W | 5.83 [5.64, 6.24] | 6.9 | −16 % |
| | D | 1.56 [1.53, 1.92] | **0.8** | **+95 %** |
| | H | 2.26 [1.82, 2.26] | 1.8 | +26 % |
| Bruno GMAW (110 A steady, 6 mm/s, 6 mm plate) | W | 5.89 [5.59, 5.93] | **4.33** | **+36 %** |
| | H | 1.99 [1.99, 2.34] | 2.14 | −7 % ✓ |

## What each failure says

1. **Overhead — not a valid comparison yet.** The hanging pool is unsteady
   at 2.5 s: frozen fraction 0.6, H_max 3.4–3.7 mm but only 0.5 mm frozen,
   metal-in-section ratio 0.46–1.58 between draws (sagging/dripping). Needs
   a longer run and a check that surface tension holds the pool (capillary
   length ≈ 5 mm, so a 2.5 mm bead should hang).
2. **Vertical-down depth — structural.** Park's shallow penetration (0.8 mm)
   comes from the pool running ahead of the arc under gravity and cushioning
   the plate. A heat volume fixed to the torch heats the plate regardless, so
   this model cannot show it. The surface-anchored source could, but on the
   flat bead it is ~0.5 mm short of depth (no crater) — the same missing
   physics again.
3. **Bruno width — transfer of η_eff and b.** η_eff = 1.0 partly stands in for
   Park's unknown voltage; Bruno's voltage is measured, so carrying η_eff over
   overheats it. The Goldak half-width (3.0 mm) was set from Park's bead, not
   from an arc-radius law, so it does not shrink for the other process.
   (Bruno's reference is the external bead width; compared with the fusion
   width at the plate face.)

## What would make it a validation

In order of leverage:

1. **Crater physics** (docs/PARK_PHYSICS_CHECK.md): the surface-heated model
   must hold the arc root near the plate face. With that, the heat source can
   follow the free surface (needed for vertical-down) and the Goldak volume
   and η_eff > 0.88 can go.
2. **Process-transferable inputs**: arc radius from a σ(I, V, CTWD) law and
   η from calorimetry, not from the calibration bead.
3. **Steady, repeated measurements**: ≥ 2 repeats per point, frozen
   fraction > 0.9 in the window (overhead needs > 2.5 s).
4. **More measured beads** at other currents/speeds of the same process (the
   user's own MIG beads) so a fit can be tested at a second process point.

Held-out data have now been seen: any change motivated by these numbers must
be re-validated on beads not used here (e.g. the Pioneer M1 wall, or new
measurements).
