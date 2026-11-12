# Conduction-only depth grid check — pre-registration
(written 12 Nov 2026, before any counted run; one smoke run at dx 0.5 with default
dt is listed as such: W 4.01, D 2.11 mm)

## Why
On the server the full model's depth keeps falling with refinement
(0.725 → 0.600 → 0.530 mm at dx 0.4/0.3/0.2, −27 %), while width and height have
flattened (docs/SERVER_MESH_PREREG.md, S3 interim). This check asks whether the
depth drift comes from the heat source / conduction / depth measurement (present
with no flow) or from the flow, free surface and deposit (absent here).

## Settings
Job `runs/steady/diag/run04_conduction.yaml`: run04 with `physics_tier: thermal`,
every flow / free-surface / force feature off, wire feed 0 (no droplets, no bead).
Same η 0.819, b 2.42 mm, Goldak torch anchor, 3.5 s, window x 24–28 mm and
`--u-design 2.0 --tau-min 0.5002` time step as the mesh study. Laptop GPU only.
Grids dx 0.5, 0.4, 0.3, 0.25, 0.2 (0.15 does not fit the 4 GB card). Deterministic
(no droplets), so one run per grid. Absolute depth is NOT comparable with the full
model (no bead, all heat to the plate); only the grid trend is used.

## Rule (on this sequence alone; no cross-platform comparison)
Depth change dx 0.4 → 0.2, conduction only:
* **|ΔD| ≤ 5 %** → source, conduction and measurement are grid-independent; the
  full model's depth drift comes from the flow / free surface / deposit.
* **|ΔD| ≥ 15 %** → the source / conduction / measurement drift on their own;
  the depth drift is (at least largely) not a flow effect.
* between → both contribute; reported as such.
Width reported alongside. Celik GCI on 0.3/0.25/0.2 if monotonic.

Results reported whatever they are.
