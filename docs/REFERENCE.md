# waam_twin reference

Detailed reference moved out of the README (11 Nov 2026). HPC runs: [HPC.md](HPC.md).

## Notebook / Colab workflow

The production notebook is `notebooks/cloud_production_workflow.ipynb`.

Use it when you want:

- Google Drive persistence
- in-run VTK sequence export
- editable job/path/preset cells without touching tracked example files

For **cluster GPU beads**, prefer the HPC copy-paste section above.

The notebook keeps three editable text blocks in memory, then writes them into a session workspace inside the repo:

| Notebook variable | Writes to | Purpose |
|------------------|-----------|---------|
| `JOB_YAML` | `jobs/notebook_session/job.yaml` | full run configuration |
| `TORCH_PATH_CSV` | `jobs/notebook_session/torch_path.csv` | welding path waypoints |
| `PRESETS_YAML` | `jobs/notebook_session/presets.yaml` | optional preset overrides for that session |

`apply_job_config()` materializes those cells and returns the `job` used by
`run_job(...)` and `run_job_live_view(...)`.

Current notebook assumptions:

- install from the repo root with `pip install -e .`
- presets are hardware/VRAM profiles only; geometry stays in `JOB_YAML`
- `strict_mode: true` is recommended for long cloud runs
- session jobs are schema-validated when loaded
- `PRESET_OVERRIDE = None` means **use `simulation.preset` from the job** (not the `WAAM_PRESET` env default)

Main notebook run controls:

| Variable | Meaning |
|----------|---------|
| `PRESET_OVERRIDE` | runtime preset override without editing the job YAML |
| `N_STEPS` | total simulation steps for a batch run (**see warning below**) |
| `EXPORT_BUNDLE` | write a final research bundle |
| `EXPORT_SEQUENCE` | export time-series VTK frames during the main run |
| `SEQUENCE_EVERY` | export every N steps when sequence export is enabled |
| `RESET_TAICHI_EACH_RUN` | reinitialize Quadrants before each run |
| `FREE_VRAM_AFTER_RUN` | release memory after the run completes |
| `KEEP_TWIN_IN_MEMORY` | keep the last twin alive for inspection/debugging |
| `SYNC_TO_DRIVE` | copy outputs to mounted Google Drive |
| `SYNC_EACH_EXPORT` | sync after each export instead of only at the end |

**`N_STEPS` warning:** the torch advances at real travel speed using fixed LBM
`dt` (`dt = 0.1 · dx`). A fixed step count that is too small **ends mid-bead**.
Steps needed ≈ `path_length_m / (travel_speed_m_s · dt)`. On HPC, prefer
`--n-steps auto` / `run_path(..., n_steps=None)` so the path is fully covered.

Path behavior is intentionally consistent:

- local CLI/Python runs usually point at `jobs/examples/...`
- notebook runs usually point at `jobs/notebook_session/...`
- both are resolved relative to the `waam_twin` repo root

## Interactive viewer

Live **PyVista** view of the bead as it is laid down. Needs
`pip install -e ".[export]"` (Quadrants has no GGUI).

```bash
# From FYP22-01 parent (PYTHONPATH=.) or any parent of waam_twin/
# Prefer: pip install -e . inside waam_twin/ (then PYTHONPATH is optional)
export PYTHONPATH=.
export WAAM_BACKEND=cuda   # or cpu / vulkan / amdgpu

# Default job: jobs/examples/bead_calibrate.yaml (free-surface LBM)
python3 -m waam_twin.viewer
# Faster on a GPU: compute only near metal (simulation.sparse_grid)
python3 -m waam_twin.viewer --sparse

# Nested under FYP22-01:
python3 -m waam_twin.viewer --job waam_twin/jobs/examples/bead_calibrate.yaml

# Legacy VOF jobs (bead_on_plate.yaml, ...) still open, with a warning: their
# wetting and droplet energy are known to be wrong.
python3 -m waam_twin.viewer --job jobs/examples/bead_on_plate.yaml
```

The default view is the **φ = 0.5 bead surface** coloured by **deposit**: wire
metal copper, plate grey, fusion-zone mixing in between, liquid orange → white.
The camera follows the torch and the top of the bead under it. Physics steps
per frame are paced automatically (`--frame-time`, default 0.3 s of physics per
frame); `+`/`-` switch to manual, `0` back to auto. The HUD shows the rate in
simulated ms per wall second.

| Key | Action |
|-----|--------|
| `M` | Cycle colour: Deposit / Temperature / HAZ / Velocity / Vorticity / Body force |
| `N` | Toggle φ = 0.5 bead surface (default) vs cell particles (`--particles` to start with them) |
| `X` / `Y` / `Z` | Section plane on that axis (same key again = off). Everything on one side is kept and the cut face is filled in (deposit or temperature colours). Across the travel direction it starts a pool length behind the arc; `Y` runs along the bead centreline; `Z` sits at the plate face (plan view of the fusion footprint) |
| `[` `]` / `{` `}` | Move the plane 1 cell / 1 mm |
| `K` | Plane rides with the torch ↔ stays fixed in the world (watch the arc arrive at a fixed station) |
| `\` | Keep the other side (by default the side away from the camera, so you face the cut) |
| `Tab` | Tilt the plane with the mouse (drag the plane or its arrow; `Tab` again to finish) |
| `C` | Camera follows the torch on / off |
| `V` | Cycle flow overlay: off / velocity arrows / streamlines |
| `B` / `H` / `F` | Particle filter: all metal / solid-only / surface (φ band) |
| `T` / `O` | Toggle porosity tracers / torch marker |
| `G` | Full research VTK bundle → `viewer_output/bundle_step_*/` |
| Right-click | Add a probe on the metal under the cursor (a click on the cut face puts it inside the part) |
| `I` / `U` | Add a probe at the screen centre / 1 mm under the metal surface below the torch |
| `L` | New probes fixed in the world (thermocouple) ↔ riding with the torch |
| `J` / `Del` | Select the next probe (its T(t) graph) / remove it |
| `P` | Screenshot PNG → `viewer_output/` |
| `R` | Reset simulation |
| `+` / `-` / `0` | More / fewer steps per frame (manual) / back to auto |
| `SPACE` | Pause / resume |
| `ESC` | Exit |

HUD **Bead** line: height, width and toe angle measured on one cross-section:
at the section plane when it cuts across the travel direction, otherwise
where the torch was 0.6 × the pool length (≥ 3 mm) ago. Height is the metal above the
plate face (sub-cell, liquid included, every layer of a wall); width spans the
columns joined to the path centreline, so spatter is ignored; the toe angle is
a coarse one-cell slope at the bead edges. Full telemetry (pool W/D, peak T,
mass balance) refreshes about once a second.

**Probes.** Each probe line on the HUD shows the current temperature, the peak,
and t8/5 (time to cool from 800 to 500 °C on the last pass through 800 °C).
The output folder holds `probes.csv` (sampled every 1 ms of simulated time),
`probes_summary.csv` (peak, t8/5, time above solidus, cooling rate at 700 °C
per probe) and `probes.yaml`, a `probes:` block you can paste into a job so a
probe set found in the viewer becomes repeatable. Job probes accept a sensor
model for comparing with real thermocouples:

```yaml
probes:
  z_ref: plate_bottom          # thermocouples drilled from underneath
  sample_every_ms: 1.0         # default 1; 0 = every step
  points:
    - {name: tc_2mm, x_mm: 35, y_mm: 22, z_mm: 7.0,
       bead_diameter_mm: 0.5,       # average over the junction bead
       time_constant_ms: 20,        # first-order sensor lag
       position_tolerance_mm: 0.5}  # also record min/max T at ±0.5 mm
```

With a sensor model the CSV gains `<name>_T_tc_K` (what the thermocouple would
read; the summary uses it) and, with a tolerance, `<name>_T_lo_K` /
`<name>_T_hi_K`.

### Resolution tuning

Two separate knobs: **simulation grid** (physics) vs **particle size** (display only).

| What | Where | Effect |
|------|--------|--------|
| **Job domain / plate** | Job YAML `simulation.domain_mm`, `plate.size_mm` | Physical experiment size (sacred). Presets have **no** domain. |
| **Hardware profile** | Job `simulation.preset` or viewer `--preset` | Collision model and tracer count — does **not** rewrite plate or `dx` |
| **Profile definitions** | [`config/presets.yaml`](../config/presets.yaml) | `vram_budget_mb`, `max_cells`, `target_dx_mm` only |
| **Particle “ball” size** | CLI | `--particle-scale 0.25` (fraction of cell width; default `0.35`) |

Example — same 50×50 plate, cheaper hardware (coarser dx):

```bash
python3 -m waam_twin.viewer \
  --job jobs/examples/bead_on_plate.yaml \
  --preset minimal \
  --particle-scale 0.28
```

Or edit the job file:

```yaml
simulation:
  preset: standard   # hardware profile (cell budget), not coupon size
  domain_mm: [80, 80, 25]
  enable_vof: true
plate:
  size_mm: [50, 50]  # geometry lives here — not in presets.yaml
```

**dx target:** `max_cells` + `vram_budget_mb` (cell-count / memory). **Not** viewer FPS. See [docs/HARDWARE.md](HARDWARE.md).

## VTK export & ParaView

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
- Legacy `.vts` paths are rewritten to `.vti`. Missing PyVista raises unless you pass `allow_skip=True` (or `run_batch.py --no-vtk`).

Full field inventory (names, units, computation): [docs/VTK_EXPORT.md](VTK_EXPORT.md).

## Bead geometry physics (job flags)

Example [`jobs/examples/bead_calibrate.yaml`](../jobs/examples/bead_calibrate.yaml)
(`physics_tier: full` — VOF/CSF/wetting/gravity/freeze + Lorentz/gas shear + Lin–Eagar
arc pressure; `enable_recoil: true` is explicit with soft-onset CC vapor recoil).
Fitted knobs (η, Goldak axes, `evap_cooling_scale`, `recoil_accommodation`) vs
predicted closures (Lin–Eagar \(p_0\), CC \(P_\mathrm{sat}\), Marangoni \(d\gamma/dT\))
are listed under the job `claim:` block. Mesh fingerprint (`dx_mm`, collision path,
`auto_dt_ma`, force/Ma caps) is locked by `assert_physics_lock` — **shop** and **HPC**
tiers do not share fitted knobs.

Held-outs must not retune those knobs:

| Job | Role |
|-----|------|
| `bead_calibrate_heldout_fast.yaml` / `_hot.yaml` | Process trends only (no absolute macro) |
| `bead_calibrate_heldout_macro2.yaml` | Park 2019 **overhead** published W/D (soft — position not modeled) |
| `bead_bruno_gmaw.yaml` | Figshare surface W/H (soft material match) |
| `wall_pioneer_m1.yaml` | Zenodo PIONEER M1 multipass |

```bash
python3 -m waam_twin.tools.prediction_report --tier shop --with-bruno
python3 -m waam_twin.tools.prediction_report --tier hpc
```

Published macro catalog (DOIs, process, twin jobs):
[docs/validation/data/published_macrograph_catalog.json](validation/data/published_macrograph_catalog.json).  
Reference write-up: [docs/validation/reference_case_ER70S6.md](validation/reference_case_ER70S6.md).  
Plain-language guide: [docs/validation/UNDERSTANDING_CALIBRATION_AND_DATA.md](validation/UNDERSTANDING_CALIBRATION_AND_DATA.md).  
HPC mesh-lock notes: [docs/HPC.md](HPC.md).  
Frozen-knob mesh band (equal travel; clamps=fail): `python -m waam_twin.tools.mesh_convergence_report`.  
Clamp probe (equal travel): `python -m waam_twin.tools.diagnostics.clamp_check --job … --travel-mm 2.0`.  
HPC clamp clearance: [`docs/HPC_CLAMP_CLEARANCE.md`](HPC_CLAMP_CLEARANCE.md).  
Recoil evidence: `python -m waam_twin.tools.recoil_accommodation_sweep`.  
Toggle sensitivity: `python -m waam_twin.tools.sensitivity_sweep`.  
Ansys Tier A grounding playbook: [`docs/validation/ansys_2024r2/PLAYBOOK_TIER_A.md`](validation/ansys_2024r2/PLAYBOOK_TIER_A.md).

| Flag | Effect |
|------|--------|
| `physics_tier` | `flow` / `full` (aliases: `base`, `default`, `standard_physics` → `flow`) — sets force defaults; later `enable_*` keys override |
| `enable_wetting` | Contact-angle CSF at substrate triple line |
| `enable_hydrostatic_gravity` | ρg flattening of liquid crest |
| `enable_bead_freeze` | Solidify cooled metal behind arc (bead crown) |
| `enable_recoil` | Vapor recoil pressure on pool surface |
| `enable_lorentz` | MHD body force (J×B) |
| `enable_gas_shear` | Shielding-gas traction on free surface |
| `enable_droplet_impact_pressure` | Droplet momentum pulse on impact |
| `enable_ctwd` | Stick-out I²R wire preheat (open-loop CTWD) |
| `enable_enthalpy_cap` | Soft T/H ceiling (`arc_physics.T_vapor_cap_K`) — safety net, not boiling physics |
| `enable_evaporative_cooling` | Hertz–Knudsen evaporative enthalpy sink on hot liquid / free surface (preferred over living on the hard cap) |

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

`transfer_mode` changes detachment period, drop mass, and impact speed heuristically; the trailing-solidify settings help clamp the far wake back to solid once it cools below a mild superheat margin. See the **Jobs & materials** section for all deposition and wetting keys.

---

## Jobs & materials

**Job YAML** lives either under `jobs/examples/` for tracked reference cases or under `jobs/notebook_session/` for notebook-generated session jobs. It defines the simulation preset, material, process sheet, optional path/frame wiring, and reference metadata.

### Top-level keys

| Key | Meaning |
|-----|---------|
| `simulation` | solver preset, `physics_tier`, and physics feature flags |
| `plate` | substrate thickness / footprint (decoupled from full domain size) |
| `material` | path to the material YAML |
| `calibration` | optional `materials/calibration/*.yaml` overlay applied after base load |
| `heat_source` | source model name such as `gaussian2d` or `goldak` (`goldak3d` is an alias) |
| `goldak` | Goldak double-ellipsoid parameters when `heat_source: goldak` |
| `arc_physics` | σ, penetration, vapor-cap, arc-pressure model |
| `advanced_physics` | Lorentz, gas-shear, conductivity, and vapor-model tuning |
| `process` | current, voltage, travel speed, wire feed, droplet transfer, ambient |
| `deposition` | droplet superheat, placement footprint, trailing solidification controls |
| `surface_wetting` | contact-angle override |
| `heat_loss` | convection/radiation boundary loss settings |
| `electrical` | CTWD / stick-out electrical properties |
| `torch_path` | inline waypoint list in mm |
| `torch_path_csv` | CSV waypoint file |
| `frame` | robot/workcell frame mapping YAML |
| `layer_height_mm` | nominal CTWD / layer rise hint (does **not** auto-offset path Z) |
| `interpass` | cooling/travel settings between passes |
| `reference` | experimental comparison targets (documentation/reporting) |
| `model_reference` | expected simulator envelope used by local regression checks |
| `probes` | named sample points. Fixed: `x_mm`, `y_mm`, `z_mm`. Torch-relative: `behind_mm` (opposite travel), optional `lateral_mm` (left of travel), and `z_mm` as a fixed height |

### `plate:`

| Key | Meaning |
|-----|---------|
| `thickness_mm` | Substrate solid height (clamped so air remains above for deposition) |
| `size_mm` | `[length, width]` footprint inside the domain |
| `origin_mm` | Optional lower-left of the plate in domain mm |

### `simulation:`

| Key | Meaning |
|-----|---------|
| `preset` | named grid preset from `config/presets.yaml` |
| `backend` | preferred backend hint; runtime still follows `init_taichi` / `WAAM_*` env |
| `physics_tier` | `flow` / `full` — default force set before `enable_*` overrides (`base`, `default`, `standard_physics` are accepted aliases to `flow`; unknown values now raise `ValueError`) |
| `domain_mm` / `dx_mm` | optional domain and cell size (override preset sizing) |
| `auto_grid` | Kept for compatibility. `dx_mm` is always allocated as requested; there is no preset VRAM/`max_cells` refusal. The device OOMs if the mesh does not fit. Cell count scales as `1/dx³`. |
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
| `enable_moving_window` | slide the local mesh with the torch (+X, ±Y; ±Z if `use_torch_z`) |
| `enable_alloy_mixing` | lerp wire/plate `alloy_frac` in liquid neighbours (default off; birth tags stay) |
| `enable_ctwd` | enable wire stick-out / CTWD preheat model |
| `use_torch_z` | use path/frame Z to update torch standoff |

### `process:`

| Key | Meaning |
|-----|---------|
| `current_A` | welding current in amperes (average / RMS-equivalent) |
| `voltage_V` | arc voltage (average / RMS-equivalent); arc power = V·I·PF·duty·η |
| `power_factor` | optional, default 1.0 — for AC processes where V·I overstates real power |
| `duty_cycle` | optional, default 1.0 — arc-on fraction for pulsed transfer |
| `arc_efficiency` | fraction of electrical power deposited as heat |
| `arc_sigma_mm` | Gaussian arc heat-source radius (default 2.0 mm; also `arc_physics.sigma_mm`) |
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

- Electrical power stored as `Q_w = V·I·PF·duty`; heat into metal is \(\eta\cdot Q_w\) per step
- if `droplet_freq_hz` is omitted, the loader infers it from `wire_feed_m_min / droplet_length_mm`
- `current_A`, `voltage_V`, and `arc_efficiency` together set the thermal input level; changing one of them can affect both pool size and bead shape

### `goldak:`

Axes follow **Goldak (1984)** naming (not every textbook figure’s lettering):

| Key | Axis |
|-----|------|
| `a_front_mm` / `a_rear_mm` | travel direction (\(x\)) |
| `b_mm` | transverse half-width (\(y\)) |
| `c_mm` | depth (\(z\)) |

| Key | Meaning |
|-----|---------|
| `ff` / `fr` | front / rear heat fractions (loader enforces \(f_f+f_r=2\)) |
| `a_front_mm` / `a_rear_mm` | travel-direction semi-axes |
| `b_mm` | transverse semi-axis |
| `c_mm` | depth semi-axis |
| `depth_front_mm` / `depth_rear_mm` | legacy → used as \(c\) if `c_mm` omitted |
| `sigma_scale` | scales \(a,b\) from arc σ when explicit axes omitted |

Geometry is **not** auto-scaled with power: larger \(Q_{\mathrm{net}}\) raises peak \(q\) on the same footprint. Enlarge axes (or lower power density) to avoid vapor-cap pinning.

### `arc_physics:`

| Key | Meaning |
|-----|---------|
| `sigma_mm` | Gaussian / pressure footprint radius (mm) |
| `penetration_mm` | arc penetration attenuation depth |
| `T_vapor_cap_K` | enthalpy/T ceiling when `enable_enthalpy_cap` is on |
| `surface_weighting` | bias heat toward free-surface metal |
| `pressure_model` | `constant` (use `pressure_pa`) or `lin_eagar` (\(p_0\propto I^2/\sigma_p^2\)) |
| `pressure_pa` | peak arc pressure when model is `constant` |
| `pressure_sigma_mm` | optional pressure σ; omit → heat σ |

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
| `recoil_accommodation` | \(C_{\mathrm{acc}}\) in \(p_{\mathrm{recoil}}=C_{\mathrm{acc}}P_{\mathrm{sat}}\) (default 0.54) |

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
| `model_reference` | expected simulator envelope for sanity checks |
| `interpass.cooling_steps` | idle cooling steps between passes |
| `interpass.travel_speed_mm_s` | non-welding travel speed during interpass motion |
| `layer_height_mm` | nominal layer rise for multi-layer jobs |

### Minimal example

```yaml
simulation:
  preset: standard
  physics_tier: flow
  enable_vof: true
  enable_csf_tension: true
  enable_wetting: true
plate:
  thickness_mm: 8.0
  size_mm: [50, 20]
material: materials/validated/ER70S-6.v1.yaml
process:
  current_A: 100
  voltage_V: 15
  arc_efficiency: 0.72
  travel_speed_mm_s: 6.5
  wire_feed_m_min: 3.2
heat_source: goldak
goldak:
  ff: 0.6
  fr: 1.4
  a_front_mm: 2.2
  a_rear_mm: 4.2
  b_mm: 3.0
  c_mm: 1.5
torch_path_csv: jobs/paths/bead_calibrate.csv
```

**Materials** are YAML under `materials/` with `status: placeholder` or `calibrated`. Placeholders print a warning at load time.

`materials.json` at the package root (if present) is **not** the simulator
authoritative source — it is a legacy/parent-UI wire list. The twin loads
`materials/**/*.yaml` via `materials.py` / `from_job()`.

See [docs/MATERIALS.md](MATERIALS.md) and [docs/HARDWARE.md](HARDWARE.md).

---

## Environment variables

| Variable | Values | Default |
|----------|--------|---------|
| `WAAM_BACKEND` | `auto`, `cpu`, `cuda`, `vulkan` | `auto` |
| `WAAM_FORCE_BACKEND` | same as above | unset (sticky; overrides `WAAM_BACKEND`) |
| `WAAM_PRESET` | `minimal`, `standard`, `high`, `ultra` | `standard` |
| `WAAM_VRAM_MB` | integer override | auto-detect |
| `WAAM_HEADLESS` | `0`, `1` | `0` |
| `WAAM_JOB` | path to job YAML | `jobs/examples/bead_on_plate.yaml` |
| `WAAM_BEAD_STEPS` | int — force bead validation step count | unset |
| `WAAM_MAX_BEAD_STEPS` | int — cap long path-derived runs | unset |
| `WAAM_FULL_VALIDATION` | `1` = process + soak + recoil smoke (not intensive / held-out) | off |
| `WAAM_BEAD_VALIDATION` | `1` = bead aspect / wetting toe / macrograph gates | off |
| `WAAM_HELDOUT_VALIDATION` | `1` = held-out prediction sims (process-only variants) | off |
| `WAAM_INTENSIVE_PHYSICS` | `1` = coupled-force / vapor / Lorentz stress tests | off |
| `WAAM_STANDARD_VALIDATION` | `1` = standard-dx pool test | off |
| `WAAM_BACKEND_MATRIX` | `1` = probe vulkan/cuda in smoke test | off |
| `WAAM_STRICT` | `1` = abort on mass/Lorentz/NaN faults | off |
| `WAAM_TRAVEL_MM` | equal-travel distance for `mesh_convergence_report` | model_reference / 8 mm |

---

## KUKA / robot integration

**Coordinate frame:** `jobs/frames/weld_table.yaml` (or `frame:` in job YAML, `WAAM_FRAME` env).

**G-code → twin path:**

```bash
# From FYP22-01 (parent on PYTHONPATH):
export PYTHONPATH=.
python3 -m waam_twin.tools.gcode_to_torch_csv part.gcode -o waam_twin/jobs/paths/part.csv
```

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

## Hardware profiles (`preset`)

Values match [`config/presets.yaml`](../config/presets.yaml):

| Profile | Typical dx start | `max_cells` / `vram_budget_mb` | Default collision |
|---------|------------------|-------------------------------|-------------------|
| `minimal` | 0.5 mm | 4e6 / 1024 | SRT |
| `standard` | 0.3 mm | 2.5e7 / 2048 | SRT |
| `high` | 0.2 mm | 1.2e8 / 8192 | TRT (two-rate) |
| `ultra` | 0.15 mm | 4e8 / 16384 | TRT (two-rate) |

Job `simulation.use_srt` overrides the preset. `use_srt: false` → TRT with
`tau_min≈0.5005` when `auto_dt_ma` is on.

`dx` is allocated as written. Cell count scales as `1/dx³`. The preset does not
coarsen `dx`. A mesh that does not fit fails at device allocation. See
[docs/HARDWARE.md](HARDWARE.md) and [docs/HPC.md](HPC.md).

**Timestep:** with `auto_dt_ma: true`, `dt` is chosen for Ma ≤ `u_mach_limit_lu`
at `u_design_m_s` and τ ≥ `tau_min` (SRT 0.505 / TRT 0.5005). Without auto Ma,
legacy acoustic scaling applies. Prefer equal **travel** (not equal steps) for
mesh studies — `WAAM_TRAVEL_MM` / `mesh_convergence_report`.

**Collision honesty note:** `use_srt: false` is **TRT / two-rate** (Ginzburg):
even modes at ω_e = 1/τ_s (viscosity), odd modes at ω_o from magic
Λ = (τ_s−½)(τ_o−½) = ¼. It is **not** a full multi-rate MRT or Geier cumulant.
An earlier Hermite-rebuild “central-moment MRT” was mass-unstable at τ→0.5005
and must not be cited. Results that still say “MRT” from before the TRT swap
should be re-run or labelled carefully.

---

## Telemetry schema

`get_telemetry()` returns a stable JSON-friendly dict: pool W/D, peak T, `n_liquid_cells`, `bead_height_mm`, `deposited_mass_g`, `mass_balance_ratio`, porosity, CTWD, toe angle, `force_diagnostics` (per-force max |F| in lattice units), …

Schema: [validation/telemetry_schema.json](../validation/telemetry_schema.json).

`strict_mode` / `WAAM_STRICT=1`: aborts on mass-balance drift, Lorentz non-convergence streaks, or NaN force diagnostics. Sustained Mach or body-force clamps warn once and keep running (populations stay synced); `mesh_convergence_report` still marks that run `FAIL_CLAMPS`.

---

