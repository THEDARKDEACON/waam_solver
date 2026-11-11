# waam_twin — calibration and validation summary
(11 Nov 2026, branch `recalibration-prep`)

## Summary
The twin is **calibrated and partly validated** for single GMAW beads on steel at a
fixed grid of dx 0.4 mm. On two independent published datasets it predicts the
direction of every process effect and most bead dimensions to 10–25 %, using one
physically plausible fitted value (arc efficiency η = 0.82–0.87). It fails where the
physics is known to be missing (no arc crater; weak power-driven spreading), and its
results are not yet shown to be grid-independent.

Every test below was pre-registered — criteria written and committed before the
runs — and nothing was re-tuned on held-out data.

## Model and what is fitted
* D3Q19 two-relaxation-time lattice Boltzmann flow with a mass-tracking free surface,
  enthalpy–porosity solidification, Marangoni (Sahoo Fe–S–O surfactant law), arc
  pressure, droplet transfer, buoyancy, Lorentz force and evaporation.
* Energy is booked honestly: arc + droplets = η·mean(U·I), with mean(U·I) from the
  measured waveform where published.
* Depth comes from a Goldak volume heat source fixed to the torch (see Limits).
* **Fitted:** η, and the Goldak half-width b where stated. **Never fitted:** material
  properties (docs/CALIBRATION.md).
* Fitting: Latin-hypercube design → Gaussian-process surrogate → Bayesian posterior
  (`tools/calib_sweep.py`).

## Test 1 — Park et al. 2019 (pulsed GMAW), measured input power
Digitising Park's U(t), I(t) figures showed the input power was 3.4 kW, not the
2.7 kW previously assumed; with it the fitted η fell from an unphysical 1.0 to
**0.87 (90 % 0.77–0.94)**. Calibration on the flat bead: W −3 %, D −5 %, H −6 %.

Held-out (docs/PARK_MEASURED_PREREG.md): **6 of 8 outputs pass.**

| bead | W | D | H |
|---|---|---|---|
| overhead | −14 % ✓ | +4 % ✓ | unsteady, not resolved |
| vertical-down | −11 % ✓ | **+132 % ✗** | +8 % ✓ |
| Bruno (other process) | **+31 % ✗** | — | +9 % ✓ |

Both failures were predicted before the run: a torch-fixed heat volume cannot show a
pool running ahead of the arc (vertical-down), and a Goldak width taken from Park's
bead does not transfer to another process (Bruno).

## Test 2 — Yanala, Andreucci & Lambiase 2026 (14 unseen beads)
Calibrated on 4 corner beads (η **0.82**, 90 % 0.80–0.84), predicted 14 others
across 2.9–3.5 kW, 450–750 mm/min, two stand-offs (docs/LAMBIASE_PREREG.md):

| check | result | limit | |
|---|---|---|---|
| penetration MAE | 22.8 % | 25 % | pass |
| height MAE | 11.8 % | 10 % | fail (narrow) |
| contact-length MAE | 20.6 % | 15 % | fail* |
| trend signs dH/dv, dCL/dv, dCL/dP | all correct | — | pass |
| η physical (0.70–0.90) | 0.82 | — | pass |
| bead width MAE (added before results*) | 14.2 % | 15 % | pass |

\*The paper's contact length exceeds its bead width in 23 of 24 rows, which its own
definition does not allow; this was found and recorded before the held-out results
were read, and a bead-width check was added alongside the original criteria.

**5 of 7 pre-registered checks pass.** The clear defect: predicted width grows about
5× too little with power.

## Hypotheses tested and rejected
Each was a pre-registered or diagnostic run aimed at the missing penetration on the
surface-heated model:
* *Numerical caps* — no force-cap hits; relaxing the speed cap changed nothing.
* *Arc crater* — arc pressure (≈ 250 Pa) plus droplet momentum (100–300 Pa) is well
  below the ≈ 900 Pa capillary resistance, and droplet kinetic energy (2 µJ) is far
  below the cavity surface energy (11 µJ): no crater forms, correctly for these forces.
* *Oxygen in the shielding gas* (docs/OXYGEN_PREREG.md) — falsified; depth does not
  respond to 0–250 ppm O.
* *Marangoni magnitude* — the force reproduces an analytic thermocapillary benchmark
  (0.047 vs 0.051 m/s).

## Limits
* **No arc crater**: penetration relies on the torch-fixed Goldak volume.
* **Weak power-driven spreading**: width–power slope ≈ 5× too low.
* **Not shown grid-independent**: all results are at dx 0.4 mm; earlier grids showed
  taller, narrower beads on refinement. A mesh study with a uniform speed ceiling is
  pre-registered and running (docs/MESH_STUDY_PREREG.md); until it reports, the
  model is a fixed-grid model and its fit does not transfer to other grids.
* **Numerical limiters**: at dx 0.4 mm the speed cap fires on part of the steps;
  relaxation time τ ≈ 0.5005; Smagorinsky eddy viscosity is on.
* **Speed**: 100–1500× slower than real time on an RTX 3050 — an offline predictor,
  not a live twin.
* Single beads only; no multilayer, residual stress or microstructure.

## Next
1. Mesh study and area-excess check (running).
2. Grid-independent contact line, then the crater/droplet impact model.
3. Re-test on new, independently measured beads (own MIG rig: 3 powers × 2 speeds,
   2 repeats, logged U·I, cross-sections), pre-registered as above.

## Sources
Park et al., Appl. Sci. 9 (2019) 4626, doi 10.3390/app9214626 ·
Yanala, Andreucci & Lambiase, Int. J. Adv. Manuf. Technol. 145 (2026) 4999,
doi 10.1007/s00170-026-18642-0 · Celik et al., J. Fluids Eng. 130 (2008) 078001 (GCI).
