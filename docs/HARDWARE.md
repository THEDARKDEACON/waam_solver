# Hardware & portability (waam_twin v2)

`waam_twin` targets **any machine** with a supported Quadrants backend.

For cluster / SLURM workflows see **[HPC.md](HPC.md)**.

## Two layers (do not conflate)

| Layer | Owns | Examples |
|-------|------|----------|
| **Job YAML** | Experiment geometry & process | `plate.size_mm`, `domain_mm`, path, I/V, `physics_tier` |
| **Hardware profile** (`preset`) | Cost / cell-count cap | `vram_budget_mb`, `max_cells`, `target_dx_mm`, SRT |

`config/presets.yaml` has **no** `domain_mm`. CLI `--preset minimal` only switches the
hardware profile and may **coarsen `dx`** until the grid fits. Plate size is
unchanged.

If the job omits `simulation.domain_mm` but sets `plate.size_mm`, the domain is
**derived**: `plate + 2×domain_margin_mm` in XY and `thickness + air_gap_mm` in Z.

## What targets dx coarsening? (not FPS)

```text
cost ≈ N_cells × physics_tier × steps
```

`auto_grid` coarsens `dx` until **both** fit:

1. `vram_budget_mb` × 0.85 (field-accurate estimate matching `WAAMGrid.estimated_vram_mb`, including Lorentz/VOF/export when the job will allocate them)
2. `max_cells` (explicit compute throttle)

The planner sizes against the **full optional field set** when `physics_tier` is `full` or Lorentz/VOF flags are on (those fields are often allocated after the grid exists). Prefer trimming `domain_mm`, enabling `enable_moving_window` (+X / ±Y / ±Z), or using `high` with a realistic `dx` before filling the card.

Viewer **FPS is not a control target**. FPS is a side effect of cell count,
physics tier, and steps-per-frame. Empty GPU VRAM with `minimal` is expected.

## Backends (auto-detect order)

1. **CUDA** — NVIDIA GPU (fastest on NVIDIA)
2. **amdgpu** — AMD GPU via ROCm/HIP
3. **Vulkan** — cross-vendor GPU (AMD/Intel fallback)
4. **CPU** — fallback for laptops without GPU

```bash
export WAAM_BACKEND=cuda   # or amdgpu, vulkan, cpu
```

## Hardware profiles

Values match [`config/presets.yaml`](../config/presets.yaml):

| Profile | Typical use | `vram_budget_mb` | `max_cells` | `target_dx_mm` |
|---------|-------------|------------------|-------------|---------------|
| `minimal` | local smoke | 1024 | 4e6 | 0.5 |
| `standard` | Default development | 2048 | 2.5e7 | 0.3 |
| `high` | Workstation GPU | 8192 | 1.2e8 | 0.2 |
| `ultra` | Large VRAM (≥16 GB) | 16384 | 4e8 | 0.15 |

```bash
export WAAM_PRESET=standard
python3 -m waam_twin.viewer --job jobs/examples/bead_on_plate.yaml --preset minimal
```

## Timestep

```text
dt = dx · u_ref_lu / u_ref_phys · dt_scale = 0.1 · dx · dt_scale   [s]
```

Default `dt_scale` is 1.0. Optional `simulation.dt_scale` multiplies `WAAMGrid.dt`
at construction (from_job / constructor). The existing **τ > 0.5** and **α_lu ≤ 1/6**
guards still abort if the scaled step is unstable. Accuracy knobs remain domain,
`dx`, and `physics_tier`. Run duration is controlled by **step count** (`n_steps` /
notebook `N_STEPS`), which must cover
`path_length / (travel_speed · dt)` or use `n_steps=None` / `--n-steps auto`.

## Moving window (+X, ±Y, ±Z)

`enable_moving_window: true` keeps a fixed local mesh and slides it with the
torch: **+X** as the bead advances, **±Y** when the torch nears a Y face
(weaving / wide walls), **±Z** when `use_torch_z` is set and the torch nears
the top or bottom of the grid (multilayer). New strips are filled with ambient
plate or gas. Tracer positions shift on the GPU with the mesh. Telemetry
reports `window_offset_{x,y,z}_mm`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAAM_BACKEND` | auto | `cuda`, `amdgpu`, `vulkan`, `cpu` |
| `WAAM_PRESET` | `standard` | Default hardware profile name |
| `WAAM_VRAM_MB` | from profile | Override grid memory/compute budget |
| `WAAM_HEADLESS` | `0` | With `--no-vtk` / `allow_skip=True`, skip VTK writers. Missing PyVista raises unless skip is requested. |
| `WAAM_MATERIAL` | — | Default material YAML path |

## Local validation

Prefer small job domains (or plate-derived domains) for smoke tests. Do not rely
on `--preset minimal` to shrink geometry.

There is no GitHub Actions workflow. Run the core suite locally:

```bash
cd FYP22-01
WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
```

FULL (`WAAM_FULL_VALIDATION=1`) and bead + held-out gates
(`WAAM_BEAD_VALIDATION=1`, `WAAM_HELDOUT_VALIDATION=1`) are the same entry
point with extra flags. Optional pytest (same core list):
`pytest waam_twin/validation/test_pytest_bridge.py`.
