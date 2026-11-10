# Park 2019 physics-core check (Nov 2026)

Question: with the arc modelled the way Park et al. (Appl. Sci. 2019,
doi 10.3390/app9214626) actually ran it, does the twin's physics core produce
the flat P-GMAW bead (fusion W 6.1, D 1.8 mm; bead H 2.2 mm) **without** a
tuned volumetric heat source?

Short answer: **no, not yet.** Width and height come out about right, but
penetration stalls at ≈1 mm because the pool surface under the arc is not
depressed. The calibrated model therefore keeps a Goldak volume for the depth
(`jobs/examples/bead_park_goldak.yaml`); see "Calibrated model" below.

## What the old lock was doing

`bead_calibrate.yaml` matched Park at dx 0.4 mm (W 5.7, D 1.6, H 2.3) through
two compensations:

1. droplet enthalpy (~1 kW) added **on top of** η·V·I from the arc
   (~+60 % energy);
2. a Goldak volume 2.6 mm deep doing the digging.

## Fixes made for the check (all committed)

| Change | Effect |
|---|---|
| `energy_split`: arc gives up a running mean of the booked droplet power | arc + droplets = η·mean(U·I) (energy budget closes on GPU) |
| pulsed mean power `V·I_mean + R(I_rms² − I_mean²)`, R = `arc_slope_ohm` 0.05 Ω (IEC 60974-1 MIG/MAG load line) | Park: 2.73 kW, not V·I_mean = 2.01 kW |
| `heat_source: surface` (column-exact Gaussian flux) | no volumetric heating |
| I_rms (150.5 A) drives I² loads (arc pressure, J×B, stick-out) | — |
| deposition carry | mass balance follows the wire feed |
| **`surface_height_at` warp race** (atomic max) | see below |

### The surface-height race (affects every earlier result)

`surface_height_at` scanned the column in the kernel's parallel loop with a
plain `k_surf = k`; on the GPU it returned the first k of whichever warp wrote
last (0, 32, 64…), never the surface. At dx 0.4 that was 0 → the plate-top
fallback, so the arc, deposition box and arc pressure always sat on the plate
face, not the bead. At dx 0.2 it returned 32 inside a 45-cell plate and
droplets stopped landing after ~0.6 s. Mesh-study results at 0.3/0.25 mm and
all earlier locks used the wrong arc_k. Regression: `validation/test_arc_input.py`.

## Results (steady bead, slices x 22–26 mm)

dx 0.4 mm unless stated; all after the race fix.

| Run | W | D | H frozen |
|---|---|---|---|
| Park measured | 6.1 | 1.8 | 2.2 |
| old lock `bead_calibrate` (double energy + Goldak) | 5.59 | 1.31 | 2.24 |
| split + **surface** σ 2.0 | 5.40 | 0.45 | 2.52 |
| same at **dx 0.2** (speed ceiling 1.75 m/s) | 5.52 | 0.35 | 2.99 |
| surface, S 100 / 150 ppm | 5.33 / 5.64 | 0.50 / 0.52 | 2.68 / 2.52 |
| surface σ 1.5 / 1.25 / 1.0 mm | 5.96 / 5.77 / 5.49 | 0.81 / 0.83 / 0.95 | 2.72 / 2.58 / 2.36 |
| split + **Goldak** (b 2.8, c 2.6) | 5.16 | 1.18 | 2.51 |
| Goldak, S 100 / 150 ppm | 5.05 / 5.47 | 1.24 / 1.10 | 2.44 / 2.61 |

Ruled out as the cause of the shallow surface-heated pool:

* **mesh / speed cap** — dx 0.2 is not deeper than 0.4;
* **Marangoni direction** — sulphur 28 → 150 ppm moves D < 0.1 mm;
* **energy sinks** — the vapour cap removes ~530 J only in the first 0.2 s
  (torch starts on the plate edge), then 0–16 W; evaporation < 5 W;
* **arc radius alone** — σ 1.0 mm still gives D 0.95.

## Why: no crater under the arc

Rosenthal conduction with Park's energy, all delivered at the plate face:
σ 2 mm → D ≈ 1.6 mm; point source → 2.4 mm. With only the arc share
(~1.2 kW; the droplet heat mostly stays in the bead metal) and σ 2 mm,
conduction gives D ≈ 0.5 mm — what the twin produces.

Measured in the twin (σ 1.25, dx 0.4, after 1 s): the free surface under the
arc sits **+1.2 mm above the plate face** on average (+0.4 … +1.6), with
liquid only 0.4 mm into the plate. The arc heats a mound of fresh metal. In
real P-GMAW the surface under the arc is pushed down to about the plate face.

The model's digging forces are weak at this current:

* Lin–Eagar arc pressure at I_rms 150 A, σ 1.25 mm: p0 ≈ 460 Pa
  (total μ0·I²/2π ≈ 4.5 mN) — comparable to the capillary pressure of a
  millimetre-scale mound (γ/R ≈ 600 Pa), so it cannot hold a crater;
* droplet impact speed from `droplet_impact_velocity_m_s`: **0.50 m/s**
  (crude feed + free-fall + pinch formula, not measured). Pulsed-GMAW droplet
  impingement is the main penetration mechanism at this current; its speed
  should come from measurement (high-speed video) or a plasma-drag model.

Also noted: `twin.droplet_freq` reads wire feed / droplet length (46.7 Hz)
while pulsed droplets fire at the pulse frequency (151.5 Hz, 4.4 mg); only the
free-fall length in the impact-speed formula uses it.

## Next physics task

Droplet impingement: impact speed from a sourced model or measurement, and
check that a 1 mm droplet at ~1 m/s, 150 Hz, opens and holds a cavity
(resolution: ≥ 5 cells per droplet diameter → dx ≤ 0.2 mm). If the surface
model then reaches D ≈ 1.8 with η and σ in their physical ranges, the Goldak
volume can be retired.

## Calibrated model

Until then: `jobs/examples/bead_park_goldak.yaml` — the honest energy split
and corrected pulsed power, with a Goldak volume whose half-width b and depth
c are fitted to the flat bead only (`jobs/calibration/sweep_park_goldak.yaml`,
η fixed 0.82). Validation is on beads the fit never saw:

* Park overhead (gravity +z) and vertical-down (gravity along travel) —
  only `simulation.welding_position` differs (new: gravity is a direction
  vector, `validation/test_hydrostatic_gravity.py`);
* Bruno et al. figshare GMAW bead — different process (steady 110 A,
  6 mm/s, 6 mm plate), bead W/H only.

Results: see `docs/PARK_VALIDATION.md`.
