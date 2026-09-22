---
title: Limitations and Open Work
step: 12
tags:
  - report/step
  - limitations
  - future-work
up: "[[WAAM Digital Twin — Project Report]]"
prev: "[[11 - Software Engineering and Tooling]]"
---

# 12 — Limitations and Open Work

> [!summary] The step in one sentence
> I would rather over-state my limitations than over-state my results, so this note collects every caveat the project documents about itself — most of them in my own words from the repository.

---

## The central limitation: one experimental anchor

Everything else follows from this. I wrote the analogy into my own validation guide and I stand by it:

> Fitting arc efficiency, Goldak shape and recoil so that **one** coupon matches is like tuning a recipe until dinner tastes right **once**. You still need other coupons to check you did not overfit.

And the consequence:

> Until the held-out coupons are filled with real cuts, absolute "we predict new welds" claims stay weak — the suite can only check **trends**.

### My own confidence table

| Use case | Confidence |
|---|---|
| Trends inside the calibrated ER70S-6 window | **Medium–high** |
| Absolute width and depth at a new current or speed, with no new cut | **Low–medium** |
| Multipass remelt and heat-affected-zone absolute numbers | **Low** until measured references are filled |
| A shop production schedule with no experiment | **Low** — overfitting risk |

---

## Gates that are not really gates

Several things that look like validation are not:

- The macrograph comparison is a **reported metric, not a pass/fail gate** in the core suite.
- Multipass gates sit behind an `awaiting_measurement` flag and do not fail.
- The held-out coupon at 5.0 mm/s reports **PENDING**, because the coupon has not been cut.
- The **Ansys 2024 R2 comparison has never been executed.** The scripts, schema and decision rubric exist. No results file does.

> [!missing] I want to be unambiguous about the Ansys package
> It is scaffolding for a comparison I intended to run and did not. Its presence in the repository should not be read as evidence of cross-code agreement.

---

## The sensitivity result I cannot yet explain

Discussed in [[10 - Validation, Calibration and Results]], repeated here because it is the most scientifically interesting open question in the project.

Sixteen toggles — Lorentz, gas shear, vapour recoil, evaporative cooling, hydrostatic gravity, droplet impact pressure, arc surface weighting, radiation, and every convection and gas-jet variation — moved predicted pool width and depth by **exactly zero** on my calibrated case. Meanwhile bead freeze, contact angle, wire feed and cell size dominated completely.

I offered three candidate explanations (genuine subdominance, a resolution artefact of the short sweep, or a wiring problem) and I cannot currently distinguish them. Resolving this is worth more than implementing another force.

> [!todo] What I would do about it
> Rerun the sweep at the full 8000 steps on the production mesh, so the baseline is the calibrated pool rather than a 2.0 mm stub, and add a numeric force-magnitude ranking table to the ablation output rather than only asserting presence and ordering. And I should write the conclusion into the prose documentation — at present it exists only in raw sweep output, which is not the same as having reported it.

---

## Resolution honesty

- The cheap Gaussian arc under-predicts macrograph pool width on coarse grids.
- The `minimal` preset at 0.5 mm cannot resolve a 7 mm pool at all; the continuous-integration tests compare against a *model* reference rather than the experiment for exactly this reason.
- The full `standard` domain is not exercised in the routine suite because of runtime.
- The capillary length for steel is about 5 mm, so wetting needs three to five cells across the toe. At the coarsest preset I am marginal, and the wetting result should be read accordingly.
- Telemetry prints width and depth to a thousandth of a millimetre. **Display precision is not tolerance**, and I say so in the README because that confusion is easy to make.

---

## Numerical caveats I have documented rather than fixed

**The collision operator is not what its filename suggests.** `cumulant_kernel.py` implements a two-relaxation-rate central-moment MRT, with higher moments projected onto equilibrium. Worse, earlier releases relaxed everything at a single rate while calling it MRT — that is central-moment SRT, and any result produced before the fix must be described that way. This is a retroactive correction to my own earlier reporting and it is in the README.

**Variable-viscosity MRT is unimplemented.** The two-rate collision uses a uniform relaxation rate; per-cell temperature-dependent viscosity works only on the single-relaxation path.

**Operator splitting is first order.** Thermal advection uses the velocity from the *end of the previous step*, so flow–thermal coupling carries a phase lag of order $\Delta t\,|\partial\mathbf{u}/\partial t|$ that grows with Marangoni number. I documented this in the timestep docstring rather than pretending the coupling is simultaneous.

**Mass balance is looser than specified.** The regression gate allows $\pm 35$ % where the specification asked for $\pm 5$ %. Strict mode enforces $\pm 5$ %, so the tight bound exists — the loose regression threshold reflects coarse-grid runs I have not yet cleaned up.

---

## Material data

Placeholder alloys print a warning at load and propagate a status field into telemetry and every metadata sidecar. I make **no accuracy claims on placeholder materials.** Even the stainless card ships as a placeholder until process calibration. A denser ER70S-6 property set exists with finer heat-capacity, conductivity and viscosity tables, but the calibrated job still runs the original card because switching would require a full re-fit and would break the physics lock.

The property provenance is honest about itself: handbook and engineering-order fits, with a note that the knots should be replaced by differential scanning calorimetry and laser-flash measurements.

> [!warning] Dataset material mismatch
> Three of the datasets I obtained are outside my calibrated window. The Bruno wire grade is uncertified in its own metadata. The PIONEER second series uses a completely different wire. The 140-bead CMT dataset uses ER100 wire on S235 plate at around 270 A — a different transfer mode and far higher heat input. I deliberately did **not** wire that last one in, because scoring it against an ER70S-6 lock would produce a number that means nothing. Held-out cases that freeze the physics knobs also freeze the **material file**.

---

## Proxies are not inspection

Porosity tracers and pool geometry are **proxies**, not non-destructive testing. My own note is blunt: closing a control loop on a wrong proxy can **create** defects. Until I have labelled experimental sessions, everything the risk layer would say stays advisory.

---

## Live mode: designed, not built

The largest single piece of unfinished work. I have a full integration plan — offline job mode as the scientific spine, live streams as an advisory twin, phased from interface freezing through recorded-bag replay to bounded supervisory control. None of the connector code exists. There is no live module in the tree.

The plan is honest about what live mode cannot be:

> It does **not** mean hard closed-loop force control at lattice-Boltzmann timestep. That is the wrong control layer for this engine.

And the arithmetic backs that up: at 0.4 mm resolution the timestep is 40 µs, so one second of process time is roughly 25 000 solver steps. A full multilayer wall at fine resolution runs well below real time.

### The eleven live-mode risks I catalogued

Time synchronisation across non-NTP clocks (more than 100 ms of skew will invent false hot spots or miss arc-off entirely). Schema and unit drift between sources. The gap between soft real-time and what an operator hears in the phrase "digital twin" — they expect one-to-one wall clock. Frame and path registration, which I have already got wrong once and put a path outside the plate. Numerical shock from abrupt process overrides, needing a low-pass filter. Shop acoustic environment, where consumer microphones miss the ultrasonic band where acoustic emission lives. False confidence in defect calls. Network fragility from hardcoded addresses. Safety authority — the twin must never override the hardware thermocouple interlayer stop or the emergency stop. Integration debt across three separate repositories with no shared package. And a missing **live parity gate**: a recorded session replayed through the twin should reproduce the offline metrics for the same path within a stated tolerance, and that test does not exist.

---

## Documentation debt

Two validation documents still publish superseded thresholds — quoting 12 % thermal and 15 % Poiseuille where the code now enforces 1 % and 8 %. They are roughly two months stale. Stale documentation that under-claims is less dangerous than documentation that over-claims, but it is still wrong.

There is also **no continuous integration.** The suite runs locally or on the cluster, by hand. For a project whose central lesson was that silent failures are the expensive ones, that is an uncomfortable omission.

---

## Consolidated to-do list

> [!todo] In roughly the order I would tackle them
> 1. **Cut the 5.0 mm/s macrograph coupon.** Single highest-value action in the project — it converts trend claims into absolute claims.
> 2. **Resolve the zero-sensitivity finding** at full resolution, and publish a numeric force ranking.
> 3. Measure two-layer remelt and heat-affected-zone extents at the calibrated operating point.
> 4. Reduce the Bruno penetration profile to a usable depth figure, or formally retire that gate.
> 5. Build an ER100 / HSLA material card so the 140-bead CMT dataset becomes usable.
> 6. Run the Ansys comparison and record the outcome either way.
> 7. Build the live connector, replay harness and risk layer, starting with recorded bags rather than live hardware.
> 8. Add the live parity gate.
> 9. Tighten mass balance from ±35 % toward ±5 %.
> 10. Tighten process gates from 25 % toward 15 % — **blocked** until item 1 provides evidence.
> 11. Refresh the two stale validation documents, and stand up continuous integration.

---

## What I would say in a viva

The twin is a **validated numerical engine with a calibrated process model**. The numerics are genuinely verified against analytical solutions — thermal diffusion to 1 %, Poiseuille to 8 %, the Stefan front to 8 %, Laplace pressure to 15 %, mass conserved to a fraction of a percent over ten thousand steps. The physics assembly has been audited defect by defect and is now guarded by regression tests that would catch each of the ten failures I found. And the process model reproduces one experimental coupon to about 7 % while predicting the correct direction of change on held-out process variants without refitting.

What it is not, yet, is an experimentally validated predictor of absolute bead geometry across a process window. That requires coupons I have not cut, and I would rather say so than dress up a single fit as a validation.

---

## Related

- [[WAAM Digital Twin — Project Report]] — back to the index
- [[10 - Validation, Calibration and Results]] — the evidence behind these caveats
- [[01 - Project Genesis and Architecture Pivots]] — the first overfit, which taught me all of this
