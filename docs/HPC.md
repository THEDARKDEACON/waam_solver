# HPC — production bead runs (waam_twin v2)

Operators: you need a **GPU** shell (or interactive GPU allocation). Two supported
paths:

| Path | When to use |
|------|-------------|
| **Docker** (below) | Preferred if your HPC allows Docker + NVIDIA Container Toolkit |
| **venv + modules** | Fallback when Docker is unavailable |

For local GGUI use the viewer; for Colab use `notebooks/cloud_production_workflow.ipynb`.

## Copy-paste: Docker (preferred when available)

From the repo root (folder with `Dockerfile` / `pyproject.toml`):

```bash
cd /path/to/waam_twin

# One-time build (reuse the image afterward)
docker build -t waam-twin:latest .

# Prove the container sees the GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

mkdir -p runs
docker run --rm --gpus all \
  -e WAAM_BACKEND=cuda \
  -v "$PWD/runs:/app/runs" \
  waam-twin:latest \
  python scripts/hpc/run_batch.py \
    --job jobs/examples/bead_on_plate_hires.yaml \
    --n-steps auto \
    --out runs/bead_on_plate_hires
```

Outputs land on the **host** under `./runs/...` (bind-mounted). Open in ParaView:

```text
runs/bead_on_plate_hires/final.pvd
runs/bead_on_plate_hires/sequence/sequence.pvd
runs/bead_on_plate_hires/bundle/
```

Useful overrides:

```bash
# Coarser smoke job
docker run --rm --gpus all -e WAAM_BACKEND=cuda -v "$PWD/runs:/app/runs" waam-twin:latest \
  python scripts/hpc/run_batch.py --job jobs/examples/bead_on_plate.yaml \
  --n-steps auto --out runs/bead_on_plate

# Lower preset inside the container
docker run --rm --gpus all -e WAAM_BACKEND=cuda -v "$PWD/runs:/app/runs" waam-twin:latest \
  python scripts/hpc/run_batch.py --job jobs/examples/bead_on_plate_hires.yaml \
  --preset standard --n-steps auto --out runs/bead_on_plate_hires_std

# Final VTK only (no sequence frames)
docker run --rm --gpus all -e WAAM_BACKEND=cuda -v "$PWD/runs:/app/runs" waam-twin:latest \
  python scripts/hpc/run_batch.py --job jobs/examples/bead_on_plate_hires.yaml \
  --n-steps auto --sequence-every 0 --out runs/bead_on_plate_hires
```

If Taichi initializes on **CPU** inside the container, the image CUDA tag likely
mismatches the host driver — rebuild from a different `nvidia/cuda:…` base or
ask the site which CUDA container tags they support.

## Copy-paste: venv (no Docker)

```bash
cd /path/to/waam_twin          # folder that contains pyproject.toml

# --- one-time (new machine / new clone) ---
module load python/3.11         # site-specific — try: module avail
module load cuda/12.2
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip && pip install -e .
nvidia-smi
python -c "import taichi as ti; ti.init(arch=ti.cuda); print(ti.cfg.arch)"

# --- every run ---
source .venv/bin/activate
# module load python/3.11 cuda/12.2   # if this is a new shell
export WAAM_BACKEND=cuda
mkdir -p runs

python scripts/hpc/run_batch.py \
  --job jobs/examples/bead_on_plate_hires.yaml \
  --n-steps auto \
  --out runs/bead_on_plate_hires
```

By default this writes **both** VTK snapshots and ParaView **`.pvd`** collections
(frame every 500 steps). Outputs:

```text
runs/bead_on_plate_hires/telemetry.json
runs/bead_on_plate_hires/final.pvd                 # end state — File→Open in ParaView
runs/bead_on_plate_hires/bundle/                   # final .vti / .vtp
runs/bead_on_plate_hires/sequence/sequence.pvd     # animation — Open → Apply → Play
runs/bead_on_plate_hires/sequence/frame_XXXX/      # per-frame VTK bundles
```

Fewer frames / less disk: add `--sequence-every 2000`. Final VTK only: `--sequence-every 0`.

### Coarser smoke (if hires OOMs)

```bash
python scripts/hpc/run_batch.py \
  --job jobs/examples/bead_on_plate.yaml \
  --n-steps auto \
  --out runs/bead_on_plate
```

| Job | Requested `dx` | Preset | Use |
|-----|----------------|--------|-----|
| `bead_on_plate.yaml` | 0.5 mm | high | Quick pipeline check |
| `bead_on_plate_hires.yaml` | 0.25 mm | high | Production-looking mesh |

`--n-steps auto` covers the full torch path. Do not replace it with a small
fixed step count or the bead may end early.

## Curbing OOM (out-of-memory) errors

If CUDA runs out of memory, reduce cost in this order:

### 1. Lower the preset

```bash
python scripts/hpc/run_batch.py \
  --job jobs/examples/bead_on_plate_hires.yaml \
  --preset standard \
  --n-steps auto \
  --out runs/bead_on_plate_hires_std
```

`ultra` → `high` → `standard` → `minimal` (each step is cheaper).

### 2. Increase `dx_mm` in the job YAML

Coarser cells ⇒ fewer cells ⇒ less VRAM. Edit your job (or a copy):

```yaml
simulation:
  preset: high
  dx_mm: 0.35          # raise from 0.25
  domain_mm: [60, 60, 22]   # optional: shrink box as well
```

### 3. Lower `max_cells` / `vram_budget_mb` in presets

Runtime reads:

```text
waam_twin/config/presets.yaml
```

(not the legacy `FYP22-01/config/presets.yaml` unless you deliberately point at it).

Example — keep profile name `high` but tighten the hard cell cap:

```yaml
high:
  vram_budget_mb: 8192
  max_cells: 40000000     # reduce if you still OOM on a 16 GB card
  target_dx_mm: 0.2
  max_tracers: 50000
  use_srt: false
```

`auto_grid` keeps the job **domain** fixed and **coarsens `dx`** until both
`vram_budget_mb × 0.85` and `max_cells` are satisfied. Watch for
`[auto_grid] … coarsened dx …` in the log.

### Extra levers

| Lever | Effect |
|-------|--------|
| `--sequence-every 0` | Skip time-series VTK (less peak RAM/disk during export) |
| Lower `max_tracers` in presets | Smaller particle buffers |
| Smaller `domain_mm` / plate | Directly fewer cells |
| `physics_tier: flow` instead of `full` | Drops Lorentz/gas-shear field cost |

The VRAM planner is optimistic; full physics can still OOM near the printed
budget — leave headroom (prefer `high` + raised `dx` before `ultra`).

See [HARDWARE.md](HARDWARE.md).

## What `module load` does

It selects the centre’s preinstalled Python/CUDA for **this shell** (`PATH`,
library paths). It does not install `waam_twin`. The venv + `pip install -e .`
does. On a laptop you usually skip `module load`.

## Why not the viewer command?

```bash
python3 -m waam_twin.viewer --job …
```

needs a GUI. HPC nodes typically have no display. Use `run_batch.py` instead;
same job YAML, headless outputs.

## If you are on a login node (no GPU)

Ask your site how to open an **interactive GPU shell**, then run the block
above. Example pattern (names vary):

```bash
srun --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Taichi prints CPU / CUDA init fails | Fix `module load cuda/…`; re-create venv after loading modules |
| CUDA OOM | See **Curbing OOM** above (preset → `dx` → `max_cells`) |
| `[auto_grid] coarsened dx` | Expected under VRAM pressure; check printed grid size |
| Job path not found | `cd` to repo root; use `jobs/examples/...` not `waam_twin/jobs/...` |

VRAM / timestep background: [HARDWARE.md](HARDWARE.md).

## Optional later

Validation suites and batch schedulers are documented for maintainers but are
**not** required for a production bead run. See the Validation section of the
README when you need them.
