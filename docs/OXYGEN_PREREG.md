# Oxygen surfactant test — pre-registration (written before any run, 10 Nov 2026)

## Hypothesis
The physical model (surface heat source, η = 0.82, arc σ 2 mm, honest energy
split) under-predicts Park's flat-bead depth (D 0.54 vs 1.8 mm) because its
surface tension ignores dissolved oxygen. Park shielded with 95 % Ar + 5 % CO₂
(paper abstract), which leaves surface-active oxygen in the pool. Oxygen
reverses the Marangoni flow to inward (downward at the centre) up to
~2140–2380 K for 100–300 ppm (Sahoo, DebRoy & McNallan 1988, Fe–O constants).
Earlier checks ruled out: speed/force caps, droplet impact speed (0.5→1.0 m/s
gave D 0.54→0.55), and droplet geometry.

## What changes
`surfactant.oxygen_ppm` in the material — competitive S+O Sahoo isotherm
(physics/surfactant.py). Default 0 reproduces the previous tables exactly.

## Fixed in advance
* Job: jobs/examples/bead_park_physics.yaml, unchanged (η 0.82, σ 2 mm, surface source).
* Oxygen: nominal 150 ppm; bounds 100 and 250 ppm (literature range for
  Ar–5 % CO₂ GMAW of C–Mn steel). Not fitted — no value outside these three is run
  for the claim.
* Window: 2.5 s, x 22–26 mm, 2 repeats at the nominal value.

## Pass criteria (flat bead, calibration-free)
W, D, H_frozen each within 15 % of Park (6.1 / 1.8 / 2.2 mm) at the nominal
oxygen, and the measured values inside the 100–250 ppm band.

## Held-outs
Park overhead / vertical-down and Bruno were seen in the Goldak validation, so a
pass on them now is a consistency check, not an independent validation. They
are run once each at the nominal value, unchanged. Bruno's gas is recorded in
its job; if it is not Ar–CO₂/O₂ the oxygen input changes accordingly from
the gas, not from the result.

## Bruno oxygen (fixed before any Bruno run)
Bruno shielded with Ar + 2 % O₂ (job header). Oxidation potential
OP = %O₂ + ½ %CO₂: Park 2.5, Bruno 2.0. Dissolved oxygen is scaled by OP:
Bruno 150 × 2.0 / 2.5 = **120 ppm**.
