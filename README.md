# waam_twin v2

GPU-accelerated WAAM melt-pool digital twin: **Taichi LBM** + **enthalpy–porosity** solidification + **VOF** free surface.  
Version **2.0.0** — portable presets, YAML materials, validated regression suite.

This repository is **standalone**. It lives inside the broader FYP22-01 WAAM stack for development but only tracks simulator code, jobs, materials, and docs. G-code cleaning / Flask UI remain in the parent project.

---

## What this project does

Predicts melt-pool **temperature**, **liquid fraction**, **Marangoni-driven flow**, and **bead geometry** for wire-arc additive manufacturing. Bead width and depth emerge from coupled physics (not drawn in as inputs). Calibration overlays (arc efficiency η, heat-loss factor, σ scale) tune process terms against reference runs.

**Explicit non-goals:** grain structure, residual-stress FEA, powder DEM, laser ray-tracing, full G-code CAM.

---

## Repository layout

```
waam_twin/                    ← git repository root (this folder)
├── README.md
├── requirements.txt
├── paths.py                  # PROJECT_ROOT = repo root
├── platform.py               # Taichi init, presets, auto_grid
├── twin.py                   # WAAMTwin orchestrator
├── grid.py                   # SoA Taichi fields
├── kernels.py                # Taichi kernels (migrating → physics/)
├── materials.py              # YAML alloy loader
├── calibration.py            # Process overlay scalars
├── job.py                    # Job YAML loader
├── torch_path.py             # CSV / waypoint torch paths
├── kuka_adapter.py           # KUKA TCP mm → sim metres (thin bridge)
├── benchmark.py              # Pool W/D measurement helpers
├── viewer/                   # Interactive Taichi GGUI viewer
├── verify.py                 # → validation.run_all
├── config/
│   └── presets.yaml          # minimal | standard | high | ultra
├── jobs/
│   ├── examples/             # bead_on_plate, multi_bead, two_layer, …
│   └── paths/                # CSV torch paths
├── materials/
│   ├── schema.json
│   ├── placeholders/         # ER70S-6, SS316L, …
│   ├── validated/            # Calibrated alloys
│   ├── calibration/          # η, σ fits per process
│   └── user/                 # Local overrides (gitignored)
├── docs/                     # HARDWARE, MATERIALS, execution plan, validation
├── physics/                  # Modular operators (re-export kernels)
│   ├── thermal.py
│   ├── phase_change.py
│   ├── forces.py
│   ├── arc.py
│   ├── free_surface.py
│   ├── deposition.py
│   ├── bead_geometry.py
│   ├── electrical_stickout.py
│   ├── weld_forces.py
│   └── lbm.py
├── export/                   # VTK bundles, probes, ParaView PVD sequences
│   ├── vtk_io.py
│   ├── bundle.py
│   └── meta.py
├── solvers/
│   └── coupled_step.py       # Single-timestep physics order
├── validation/               # Regression tests + baselines
├── tools/
│   ├── fit_calibration.py
│   ├── run_validation_matrix.py
│   └── benchmark_performance.py
└── .github/workflows/verify.yml
```

---

## Architecture

### Data flow

```mermaid
flowchart TB
  subgraph inputs [Inputs]
    JOB[Job YAML]
    MAT[Material YAML]
    CAL[Calibration YAML]
    PATH[Torch path CSV/YAML]
  end

  subgraph platform [Platform]
    INIT[platform.init_taichi]
    PRE[presets.yaml + auto_grid]
  end

  subgraph twin [WAAMTwin]
    RESET[reset / init_grid]
    STEP[step / run_path]
  end

  subgraph solver [coupled_step]
    ARC[arc.inject_heat]
    DEP[deposition.feed_wire_surface]
    TH[thermal.advect_diffuse]
    PC[phase_change.update]
    VOF[free_surface: φ advect + wetting BC]
    FRC[forces: CSF Marangoni buoyancy gravity]
    WELD[weld_forces: recoil Lorentz gas shear droplet]
    LBM[lbm.collide + stream]
    FRZ[remelt + solidify bead_freeze]
  end

  subgraph outputs [Outputs]
    TEL[get_telemetry JSON]
    VTK[export_vtk / export_surface_vtk]
    HAZ[export_haz_vtk T_max]
  end

  JOB --> twin
  MAT --> twin
  CAL --> twin
  PATH --> twin
  INIT --> twin
  PRE --> twin
  RESET --> STEP
  STEP --> solver
  solver --> TEL
  solver --> VTK
  solver --> HAZ
```

### Physics timestep (`solvers/coupled_step.py`)

Each `WAAMTwin.step(x, y, is_welding)` runs, in order:

1. Clear body forces; optional CTWD update  
2. Arc heat injection (Gaussian2D / Goldak) + enthalpy cap  
3. Wire droplet schedule → `feed_wire_surface` + tracer inject + droplet impact  
4. Thermal advection–diffusion + boundary losses  
5. `T_max` / cooling rate; enthalpy–porosity phase update  
6. VOF φ advection, reinit, **contact-angle wetting BC**, flag update (optional)  
7. CSF tension, Marangoni, **hydrostatic gravity**, thermal buoyancy  
8. Lorentz MHD, gas shear, arc pressure, recoil (optional flags)  
9. LBM collide (SRT or MRT) + stream; tracer advection  
10. **Remelt hot solid + solidify cooled metal** (bead freeze / substrate growth)  
11. Buffer swap  

See [docs/BEAD_GEOMETRY_PHYSICS_SPEC.md](docs/BEAD_GEOMETRY_PHYSICS_SPEC.md) and [docs/weld_pool_physics.md](docs/weld_pool_physics.md).

### Layer responsibilities

| Module | Role |
|--------|------|
| `platform.py` | CUDA → Vulkan → CPU fallback; VRAM-aware grid sizing |
| `grid.py` | Ping-pong LBM distributions, T, H, φ, flags, tracers |
| `physics/*` | Thin API over `kernels.py` (ongoing migration) |
| `kernels.py` | Taichi `@ti.kernel` implementations |
| `twin.py` | Public API: `from_preset`, `from_job`, `run_path`, exports |
| `validation/` | Kernel-only and process benchmarks |

---

## Installation

**Requirements:** Python 3.11+, Linux recommended (Taichi CPU/Vulkan/CUDA).

```bash
git clone <your-remote-url> waam_twin    # folder name must be waam_twin
cd waam_twin
pip install -r requirements.txt
```

### PYTHONPATH (important)

The Python package name is `waam_twin`, so the **parent** of this directory must be on `PYTHONPATH`:

```bash
cd waam_twin
export PYTHONPATH="$(cd .. && pwd)"
python -m waam_twin.validation.run_all
```

When this repo is nested inside FYP22-01 (as during local development):

```bash
cd /path/to/FYP22-01
export PYTHONPATH=.
python -m waam_twin.validation.run_all
```

---

## Quick start

### Bead-on-plate from a job file

```bash
export PYTHONPATH="$(cd .. && pwd)"   # or . if under FYP22-01
export WAAM_BACKEND=cpu
export WAAM_PRESET=minimal

python3 -c "
from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
init_taichi()
t = WAAMTwin.from_job('jobs/examples/bead_on_plate.yaml')
t.reset()
t.run_path('jobs/examples/bead_on_plate.yaml', n_steps=600)
print(t.get_telemetry())
"
```

### Preset-only (no job file)

```python
from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin

init_taichi(backend="cpu")
twin = WAAMTwin.from_preset("standard", material="materials/validated/ER70S-6.v1.yaml")
twin.reset()
twin.step(0.015, 0.010, is_welding=True)
```

### Notebook / Colab workflow

The production notebook is `notebooks/cloud_production_workflow.ipynb`.

Use it when you want:

- long unattended runs
- Google Drive persistence
- in-run VTK sequence export
- editable job/path/preset cells without touching tracked example files

The notebook keeps three editable text blocks in memory, then writes them into a session workspace inside the repo:

| Notebook variable | Writes to | Purpose |
|------------------|-----------|---------|
| `JOB_YAML` | `jobs/notebook_session/job.yaml` | full run configuration |
| `TORCH_PATH_CSV` | `jobs/notebook_session/path.csv` | welding path waypoints |
| `PRESETS_YAML` | `config/presets.yaml` | optional preset overrides for that session |

`apply_job_config()` materializes those cells and returns the `job` used by `run_job(...)` and `run_job_live_view(...)`.

Main notebook run controls:

| Variable | Meaning |
|----------|---------|
| `PRESET_OVERRIDE` | runtime preset override without editing the job YAML |
| `N_STEPS` | total simulation steps for a batch run |
| `EXPORT_BUNDLE` | write a final research bundle |
| `EXPORT_SEQUENCE` | export time-series VTK frames during the main run |
| `SEQUENCE_EVERY` | export every N steps when sequence export is enabled |
| `RESET_TAICHI_EACH_RUN` | reinitialize Taichi before each run |
| `FREE_VRAM_AFTER_RUN` | release memory after the run completes |
| `KEEP_TWIN_IN_MEMORY` | keep the last twin alive for inspection/debugging |
| `SYNC_TO_DRIVE` | copy outputs to mounted Google Drive |
| `SYNC_EACH_EXPORT` | sync after each export instead of only at the end |

Path behavior is intentionally consistent:

- local CLI/Python runs usually point at `jobs/examples/...`
- notebook runs usually point at `jobs/notebook_session/...`
- both are resolved relative to the `waam_twin` repo root

For quick local debugging, use the normal CLI/viewer workflow below. For long production runs, prefer the notebook.

### Interactive viewer

Real-time **Taichi GGUI** particle view of the melt pool (voxel-based, not ParaView quality). Defaults to the calibrated bead-on-plate job (VOF + heat loss + calibration overlay).

```bash
cd FYP22-01   # parent on PYTHONPATH
export PYTHONPATH=.
export WAAM_BACKEND=cuda   # or cpu / vulkan

# Calibrated job (recommended)
python3 -m waam_twin.viewer --job jobs/examples/bead_on_plate.yaml

# Multi-bead path job
python3 -m waam_twin.viewer --job jobs/examples/multi_bead.yaml

# Preset-only demo (no job file)
python3 -m waam_twin.viewer --preset minimal --material materials/validated/ER70S-6.v1.yaml
```

| Key | Action |
|-----|--------|
| `M` | Cycle view: Temperature / HAZ / Velocity / Vorticity / Body force (+ force arrows in body-force mode) |
| `V` | Cycle flow overlay: off / velocity arrows / streamlines |
| `B` / `H` / `F` | Filter: all metal / solid-only / surface (φ band) |
| `N` | Toggle φ surface shell (T-colored) vs particles |
| `C` / `Z` | Toggle Y / Z cross-section clip |
| `T` / `O` | Toggle porosity tracers / torch marker |
| `G` | Full research VTK bundle → `viewer_output/bundle_step_*/` |
| `I` | Pick probe at camera lookat; **T(t)** panel updates live |
| `P` | Add probe at torch (CSV on **`G`** export) |
| `R` | Reset simulation |
| `+` / `-` | More / fewer physics steps per frame |
| `S` | Screenshot PNG → `viewer_output/` |
| `SPACE` | Pause / resume |
| `ESC` | Exit |

HUD shows pool W/D, peak T, Marangoni speed, liquid cell count, bead height, deposited mass, and a **T(t) probe** sparkline when a probe is active (`P`, `I`, or job `probes:`).

Temperature view colors hot liquid, warm HAZ, substrate (dark), and frozen bead (bronze) — not flat grey on solids.

#### Resolution tuning

Two separate knobs: **simulation grid** (physics) vs **particle size** (display only).

| What | Where | Effect |
|------|--------|--------|
| **Grid / cell size** | Job YAML `simulation.preset` or viewer `--preset` | `minimal` → dx≈0.5 mm, `standard` → 0.3 mm, `high` → 0.2 mm |
| **Preset definitions** | [`config/presets.yaml`](config/presets.yaml) | Edit `target_dx_mm`, `domain_mm`, `vram_budget_mb` per tier |
| **Viewer override** | CLI | `--preset standard` overrides job preset without editing YAML |
| **Particle “ball” size** | CLI | `--particle-scale 0.25` (fraction of cell width; default `0.35`) |

Example — finer physics + smaller particles:

```bash
python3 -m waam_twin.viewer \
  --job jobs/examples/bead_on_plate.yaml \
  --preset standard \
  --particle-scale 0.28
```

Or edit the job file:

```yaml
simulation:
  preset: standard   # was minimal — 0.3 mm cells, larger grid
  enable_vof: true
```

**VRAM:** `standard` needs ~2 GB GPU budget; `high` ~8 GB. Your RTX-class laptop can usually run `standard` on CUDA.

### VTK export & ParaView

**Single snapshot** (one time step — ParaView play will not animate):

```python
twin.export_vtk_full("pool.vti", tiers=(0, 1, 2, 3))
twin.export_surface_vtk("bead_surface.vtp")
twin.export_research_bundle("run_out/")   # volume + surface + tracers + meta + telemetry JSON
```

Press **`G`** in the viewer for a full bundle under `viewer_output/bundle_step_*/`.

**Time-series / build-up animation** — use the export CLI, then open the **`.pvd`** file:

```bash
python3 -m waam_twin.export \
  --job jobs/examples/bead_on_plate.yaml \
  --preset standard \
  --steps 5000 \
  --every 100 \
  --max-frames 50 \
  --out viewer_output/my_bead_run

# ParaView: File → Open → viewer_output/my_bead_run/sequence.pvd → Apply → Play
```

Each frame folder contains:

| File | Contents |
|------|----------|
| `volume_step_*.vti` | `Temperature_K`, `Liquid_Fraction`, `VOF_phi`, `Cell_Flags`, `T_max_K`, velocities, optional forces |
| `surface_step_*.vtp` | φ / f_l = 0.5 isosurface (bead crown) |
| `telemetry_step_*.json` | `bead_height_mm`, `deposited_mass_g`, pool W/D, … |
| `meta_step_*.json` | Grid, material, physics flags, unit conversions |

**ParaView tips**

- Open **`sequence.pvd`**, not a lone `volume_step_*.vti`.
- Turn **Z clip off** in the viewer (`Z`) before exporting if you need the full crown in screenshots.
- Threshold `Cell_Flags` (0 = fluid, 1 = solid) and clip Z above substrate to isolate deposited bead.
- Legacy `.vts` paths are rewritten to `.vti`. Set `WAAM_HEADLESS=1` to skip VTK in batch runs.

Full field inventory (names, units, computation): [docs/VTK_EXPORT.md](docs/VTK_EXPORT.md). Implementation spec: [docs/DIAGNOSTICS_AND_VTK_SPEC.md](docs/DIAGNOSTICS_AND_VTK_SPEC.md).

### Bead geometry physics (job flags)

Example [`jobs/examples/bead_on_plate.yaml`](jobs/examples/bead_on_plate.yaml):

| Flag | Effect |
|------|--------|
| `enable_wetting` | Contact-angle CSF at substrate triple line |
| `enable_hydrostatic_gravity` | ρg flattening of liquid crest |
| `enable_bead_freeze` | Solidify cooled metal behind arc (bead crown) |
| `enable_recoil` | Vapor recoil pressure on pool surface |
| `enable_lorentz` | MHD body force (J×B) |
| `enable_gas_shear` | Shielding-gas traction on free surface |
| `enable_droplet_impact_pressure` | Droplet momentum pulse on impact |
| `enable_ctwd` | Stick-out I²R wire preheat (open-loop CTWD) |

New droplet / impact knobs:

```yaml
process:
  transfer_mode: globular   # globular | spray | pulsed | auto
  droplet_freq_hz: 44.0     # optional base detachment rate
  pulse_frequency_hz: 90.0  # for pulsed transfer
  droplet_size_jitter: 0.12 # deterministic ± modulation
  impact_lead_angle_deg: 8.0

deposition:
  trailing_solidify_lookback_mm: 2.5
  trailing_solidify_temp_margin_K: 35.0
```

`transfer_mode` changes detachment period, drop mass, and impact speed heuristically; the trailing-solidify settings help clamp the far wake back to solid once it cools below a mild superheat margin.

Spec: [docs/BEAD_GEOMETRY_PHYSICS_SPEC.md](docs/BEAD_GEOMETRY_PHYSICS_SPEC.md).

---

## Jobs & materials

**Job YAML** lives either under `jobs/examples/` for tracked reference cases or under `jobs/notebook_session/` for notebook-generated session jobs. It defines the simulation preset, material, process sheet, optional path/frame wiring, and reference metadata.

### Top-level keys

| Key | Meaning |
|-----|---------|
| `simulation` | solver preset and physics feature flags |
| `material` | path to the material YAML |
| `calibration` | optional `materials/calibration/*.yaml` overlay applied after base load |
| `heat_source` | source model name such as `gaussian2d` or `goldak` |
| `goldak` | Goldak double-ellipsoid parameters when `heat_source: goldak` |
| `arc_physics` | arc penetration and vapor-cap settings |
| `advanced_physics` | Lorentz, gas-shear, conductivity, and vapor-model tuning |
| `process` | current, voltage, travel speed, wire feed, droplet transfer, ambient |
| `deposition` | droplet superheat, placement footprint, trailing solidification controls |
| `surface_wetting` | contact-angle override |
| `heat_loss` | convection/radiation boundary loss settings |
| `electrical` | CTWD / stick-out electrical properties |
| `torch_path` | inline waypoint list in mm |
| `torch_path_csv` | CSV waypoint file |
| `frame` | robot/workcell frame mapping YAML |
| `layer_height_mm` | nominal layer rise for multi-layer jobs |
| `interpass` | cooling/travel settings between passes |
| `reference` | experimental comparison targets (documentation/reporting) |
| `model_reference` | expected simulator envelope used by CI/regression checks |
| `probes` | named sample points recorded during the run |

### `simulation:`

| Key | Meaning |
|-----|---------|
| `preset` | named grid preset from `config/presets.yaml` |
| `backend` | preferred backend hint for notebooks/docs; runtime still follows `init_taichi(...)` / env vars |
| `enable_vof` | enable free-surface VOF advection |
| `enable_csf_tension` | enable capillary surface-tension force |
| `enable_wetting` | apply the contact-angle wetting boundary condition |
| `enable_hydrostatic_gravity` | add full hydrostatic `ρg` flattening |
| `enable_bead_freeze` | solidify cooled deposited metal behind the arc |
| `enable_recoil` | enable vapor recoil pressure |
| `use_recoil_clausius_clapeyron` | use the Clausius-Clapeyron recoil model |
| `enable_lorentz` | enable electromagnetic `J×B` body force |
| `enable_gas_shear` | enable shielding-gas traction |
| `enable_droplet_impact_pressure` | apply droplet impact momentum pulse |
| `enable_enthalpy_cap` | clamp extreme enthalpy / vaporization spikes |
| `arc_surface_weighting` | bias arc heating toward the top free surface |
| `enable_substrate_growth` | allow hot solid/substrate remelt and growth studies |
| `enable_moving_window` | enable moving-domain/window logic |
| `enable_ctwd` | enable wire stick-out / CTWD preheat model |
| `use_torch_z` | use path/frame Z to update torch standoff |

### `process:`

| Key | Meaning |
|-----|---------|
| `current_A` | welding current in amperes |
| `voltage_V` | arc voltage; combined with current and efficiency for nominal arc power |
| `arc_efficiency` | fraction of electrical power deposited as heat |
| `travel_speed_mm_s` | torch travel speed along the path |
| `wire_feed_m_min` | wire feed speed; drives deposition mass rate |
| `wire_diameter_mm` | filler wire diameter |
| `droplet_length_mm` | characteristic wire length per droplet when frequency is inferred |
| `T_ambient_K` | ambient/reference temperature |
| `stickout_mm` | exposed wire stick-out length |
| `ctwd_mm` | contact-tip-to-work distance |
| `transfer_mode` | transfer heuristic: `globular`, `spray`, `pulsed`, or `auto` |
| `droplet_freq_hz` | explicit droplet detachment frequency override |
| `pulse_frequency_hz` | pulse frequency for pulsed-transfer heuristics |
| `droplet_size_jitter` | deterministic droplet-size modulation factor |
| `impact_lead_angle_deg` | forward impact-angle bias relative to travel direction |

Notes:

- if `droplet_freq_hz` is omitted, the loader infers it from `wire_feed_m_min / droplet_length_mm`
- `current_A`, `voltage_V`, and `arc_efficiency` together set the thermal input level; changing one of them can affect both pool size and bead shape

### `goldak:`

| Key | Meaning |
|-----|---------|
| `ff` | front heat fraction |
| `fr` | rear heat fraction |
| `depth_front_mm` | front ellipsoid penetration depth |
| `depth_rear_mm` | rear ellipsoid penetration depth |

### `arc_physics:`

| Key | Meaning |
|-----|---------|
| `penetration_mm` | arc penetration attenuation depth |
| `T_vapor_cap_K` | temperature ceiling for vapor/enthalpy cap behavior |
| `surface_weighting` | alternate place to enable surface-weighted arc deposition |

### `advanced_physics:`

| Key | Meaning |
|-----|---------|
| `gas_jet_velocity_m_s` | gas jet speed used by gas-shear forcing |
| `gas_shear_coeff` | scale factor for gas-shear traction |
| `sigma_liquid_Sm` | liquid electrical conductivity |
| `sigma_solid_Sm` | solid electrical conductivity |
| `lorentz_jacobi_iters` | iteration count for the Lorentz solve |
| `T_boiling_K` | boiling temperature used by recoil/vapor models |
| `L_vapor_J_kg` | latent heat of vaporization |
| `R_spec_vapor_J_kgK` | vapor specific gas constant |

### `deposition:`

| Key | Meaning |
|-----|---------|
| `superheat_K` | temperature excess assigned to deposited droplets |
| `footprint_sigma_scale` | multiplier on deposition spread; lower values usually raise bead crown |
| `trailing_solidify_lookback_mm` | distance behind the arc checked for trailing freeze |
| `trailing_solidify_temp_margin_K` | margin above liquidus used as the freeze threshold |

### `surface_wetting:`

| Key | Meaning |
|-----|---------|
| `contact_angle_deg` | static wetting/contact angle used by the wall boundary condition |

### `heat_loss:`

| Key | Meaning |
|-----|---------|
| `convection` | enable convective boundary losses |
| `radiation` | enable radiative boundary losses |
| `h_conv` | convective heat-transfer coefficient |
| `eps_rad` | emissivity for radiation losses |

### `electrical:`

| Key | Meaning |
|-----|---------|
| `rho_e_ohm_m` | electrical resistivity for stick-out heating |
| `eta_stick` | fraction of resistive stick-out heating retained by the wire |

### Paths, probes, and references

| Key | Meaning |
|-----|---------|
| `torch_path` | inline list of `{x_mm, y_mm, z_mm}` waypoints |
| `torch_path_csv` | external CSV of torch waypoints |
| `frame` | frame transform YAML for robot/workcell coordinates |
| `probes` | list of named `{name, x_mm, y_mm, z_mm}` sample points |
| `reference` | experimental targets for comparison/reporting |
| `model_reference` | expected simulator envelope for sanity/CI checks |
| `interpass.cooling_steps` | idle cooling steps between passes |
| `interpass.travel_speed_mm_s` | non-welding travel speed during interpass motion |
| `layer_height_mm` | nominal layer rise for multi-layer jobs |

### Minimal example

```yaml
simulation:
  preset: standard
  enable_vof: true
  enable_csf_tension: true
  enable_wetting: true
material: materials/validated/ER70S-6.v1.yaml
process:
  current_A: 120
  voltage_V: 18
  arc_efficiency: 0.72
  travel_speed_mm_s: 8
  wire_feed_m_min: 2.5
heat_source: goldak
torch_path_csv: jobs/paths/bead_line.csv
```

**Materials** are YAML under `materials/` with `status: placeholder` or `calibrated`. Placeholders print a warning at load time.

See [docs/MATERIALS.md](docs/MATERIALS.md) and [docs/HARDWARE.md](docs/HARDWARE.md).

---

## Environment variables

| Variable | Values | Default |
|----------|--------|---------|
| `WAAM_BACKEND` | `auto`, `cpu`, `cuda`, `vulkan` | `auto` |
| `WAAM_PRESET` | `minimal`, `standard`, `high`, `ultra` | `standard` |
| `WAAM_VRAM_MB` | integer override | auto-detect |
| `WAAM_HEADLESS` | `0`, `1` | `0` |
| `WAAM_JOB` | path to job YAML | `jobs/examples/bead_on_plate.yaml` (under `waam_twin/`) |
| `WAAM_FULL_VALIDATION` | `1` = process + soak tests | off |
| `WAAM_STANDARD_VALIDATION` | `1` = standard-dx pool test | off |
| `WAAM_BACKEND_MATRIX` | `1` = probe vulkan/cuda in smoke test | off |

---

## Validation

```bash
# Core CI (~30 s)
WAAM_BACKEND=cpu PYTHONPATH=... python3 -m waam_twin.validation.run_all

# Full suite (~2 min)
WAAM_FULL_VALIDATION=1 WAAM_BACKEND=cpu PYTHONPATH=... python3 -m waam_twin.validation.run_all

# Standard cell size pool gate
WAAM_STANDARD_VALIDATION=1 WAAM_BACKEND=cpu PYTHONPATH=... python3 -m waam_twin.validation.run_all
```

**Tools:**

```bash
python3 -m waam_twin.tools.fit_calibration --write
python3 -m waam_twin.tools.run_validation_matrix --quick
python3 -m waam_twin.tools.benchmark_performance
```

Legacy entry: `python3 -m waam_twin.verify` (delegates to `run_all`).

---

## KUKA / robot integration

**Coordinate frame:** `jobs/frames/weld_table.yaml` (or `frame:` in job YAML, `WAAM_FRAME` env).

**G-code → twin path:**

```bash
# From FYP22-01 (parent on PYTHONPATH):
export PYTHONPATH=.
python3 -m waam_twin.tools.gcode_to_torch_csv part.gcode -o waam_twin/jobs/paths/part.csv
```

Flask upload (`POST /api/gcode`) also writes `waam_twin/jobs/paths/<PROGRAM>.csv` automatically.

**Job wiring:**

```yaml
frame: jobs/frames/weld_table.yaml
simulation:
  use_torch_z: true   # map robot Z + CTWD → arc height
torch_path_csv: jobs/paths/part.csv
```

**Live MockKUKA:** `kuka.py` calls `kuka_adapter.step_from_tcp()` with `$POS_ACT` mm values.

```bash
export WAAM_JOB=jobs/examples/bead_on_plate.yaml
export WAAM_FRAME=jobs/frames/weld_table.yaml
export WAAM_PRESET=minimal
```

No robot logic lives inside this package — only frame mapping, CSV paths, and twin construction.

---

## Presets

| Preset | Typical dx | VRAM budget | Collision |
|--------|------------|-------------|-----------|
| `minimal` | 0.5 mm | 512 MB | SRT |
| `standard` | 0.3 mm | 2 GB | SRT |
| `high` | 0.2 mm | 8 GB | MRT |
| `ultra` | 0.15 mm | 16 GB | MRT |

Grid dimensions are computed by `auto_grid()` from `config/presets.yaml` and available memory.

---

## Telemetry schema

`get_telemetry()` returns a stable JSON-friendly dict: pool W/D, peak T, `n_liquid_cells`, `bead_height_mm`, `deposited_mass_g`, `mass_balance_ratio`, porosity, CTWD, toe angle, …

Schema: [validation/telemetry_schema.json](validation/telemetry_schema.json).

---

## Further reading

- [Bead geometry physics spec](docs/BEAD_GEOMETRY_PHYSICS_SPEC.md) — wetting, deposition, freeze, CTWD  
- [Weld pool forces](docs/weld_pool_physics.md) — Marangoni, Lorentz, recoil, droplets  
- [VTK & diagnostics spec](docs/DIAGNOSTICS_AND_VTK_SPEC.md) — export tiers, ParaView workflow  
- [Execution plan](docs/WAAM_TWIN_V2_EXECUTION_PLAN.md) — phases, task IDs, exit gates  
- [Validation report](docs/validation/VALIDATION_REPORT.md)  
- [ER70S-6 reference case](docs/validation/reference_case_ER70S6.md)  
- [LBM numerics](docs/physics/LBM.md)

---

## Relationship to FYP22-01

| In `waam_twin/` (this repo) | In parent FYP22-01 only |
|-----------------------------|-------------------------|
| `config/`, `jobs/`, `materials/`, `docs/` | Flask UI (`main.py`), `kuka.py` |
| Job / material YAML | Parent `materials.json` (UI wire list) |
| `kuka_adapter.py` | `waam_physics.py`, `gcode_pipeline.py` |

When nested under FYP22-01: `export PYTHONPATH=.` on the **parent**. Paths like `jobs/examples/…` resolve inside **`waam_twin/`** via `paths.resolve_project_path()`.

See also [../README_WAAM_TWIN.md](../README_WAAM_TWIN.md) for a short FYP22-01 entry point.
