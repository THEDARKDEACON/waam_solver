# Hardware & portability (waam_twin v2)

`waam_twin` targets **any machine** with a supported Taichi backend.

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

1. `vram_budget_mb` × 0.85 (optimistic memory estimate + headroom)
2. `max_cells` (explicit compute throttle)

The planner uses a simplified bytes-per-cell estimate. Full-physics allocations
(Lorentz, VOF, material tables, force snaps, viewer buffers) can use more VRAM
than estimated — `ultra` on a true 16 GB card can still OOM. Prefer trimming
`domain_mm`, enabling `enable_moving_window`, or using `high` with a realistic
`dx` before filling the card.

Viewer **FPS is not a control target**. FPS is a side effect of cell count,
physics tier, and steps-per-frame. Empty GPU VRAM with `minimal` is expected.

## Backends (auto-detect order)

1. **CUDA** — NVIDIA GPU (fastest)
2. **Vulkan** — AMD / Intel / cross-vendor GPU
3. **CPU** — fallback for CI and laptops without GPU

```bash
export WAAM_BACKEND=cuda   # or vulkan, cpu
```

## Hardware profiles

Values match [`config/presets.yaml`](../config/presets.yaml):

| Profile | Typical use | `vram_budget_mb` | `max_cells` | `target_dx_mm` |
|---------|-------------|------------------|-------------|---------------|
| `minimal` | CI, interactive smoke | 1024 | 4e6 | 0.5 |
| `standard` | Default development | 2048 | 2.5e7 | 0.3 |
| `high` | Workstation GPU | 8192 | 1.2e8 | 0.2 |
| `ultra` | Large VRAM (≥16 GB) | 16384 | 4e8 | 0.15 |

```bash
export WAAM_PRESET=standard
python3 -m waam_twin.viewer --job jobs/examples/bead_on_plate.yaml --preset minimal
```

## Timestep (not user-tunable)

```text
dt = dx · u_ref_lu / u_ref_phys = 0.1 · dx   [s]
```

There is no job-YAML `dt` knob. Accuracy knobs are domain, `dx`, and
`physics_tier`. Run duration is controlled by **step count** (`n_steps` /
notebook `N_STEPS`), which must cover
`path_length / (travel_speed · dt)` or use `n_steps=None` / `--n-steps auto`.

## Moving window

`enable_moving_window: true` keeps a fixed local mesh and slides it in X as the
torch advances — useful for long beads without a huge `domain_mm`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAAM_BACKEND` | auto | `cuda`, `vulkan`, `cpu` |
| `WAAM_PRESET` | `standard` | Default hardware profile name |
| `WAAM_VRAM_MB` | from profile | Override grid memory/compute budget |
| `WAAM_HEADLESS` | `0` | Skip VTK export when `1` |
| `WAAM_MATERIAL` | — | Default material YAML path |

## CI

Prefer small job domains (or plate-derived domains) for smoke tests. Do not rely
on `--preset minimal` to shrink geometry.

```bash
cd FYP22-01
WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
```
