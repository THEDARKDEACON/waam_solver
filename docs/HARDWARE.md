# Hardware & portability (waam_twin v2)

`waam_twin` targets **any machine** with a supported Taichi backend. Grid size and tracer count are chosen from available memory, not from a fixed RTX 3050 configuration.

## Backends (auto-detect order)

1. **CUDA** — NVIDIA GPU (fastest)
2. **Vulkan** — AMD / Intel / cross-vendor GPU
3. **CPU** — fallback for CI and laptops without GPU

```bash
export WAAM_BACKEND=cuda   # or vulkan, cpu
```

## Presets

Presets in `config/presets.yaml` define domain size, target resolution, VRAM budget, and collision model:

| Preset | Typical use |
|--------|-------------|
| `minimal` | CI, smoke tests, laptops |
| `standard` | Default development |
| `high` | Workstation GPU |
| `ultra` | Large VRAM (≥16 GB) |

```bash
export WAAM_PRESET=standard
```

`WAAMTwin.from_preset("standard")` calls `auto_grid()` to fit `nx×ny×nz` within the preset VRAM budget.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAAM_BACKEND` | auto | `cuda`, `vulkan`, `cpu` |
| `WAAM_PRESET` | `standard` | Grid preset name |
| `WAAM_VRAM_MB` | from preset | Override VRAM budget |
| `WAAM_HEADLESS` | `0` | Skip VTK export when `1` |
| `WAAM_MATERIAL` | — | Default material YAML path |

## VRAM estimate

`WAAMGrid.estimated_vram_mb()` reports approximate field + tracer memory. `platform.check_vram_budget()` warns if the chosen grid exceeds the budget.

## CI

GitHub Actions runs the full validation suite on **CPU** (`WAAM_BACKEND=cpu`, `WAAM_PRESET=minimal`) so merges are not gated on GPU runners.

## Local validation

```bash
cd FYP22-01
WAAM_BACKEND=cpu PYTHONPATH=. python3 -m waam_twin.validation.run_all
```
