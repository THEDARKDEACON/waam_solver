# HPC & cluster runs (waam_twin v2)

Production guide for GPU clusters (SLURM assumed). For Colab / Jupyter, use
`notebooks/cloud_production_workflow.ipynb` instead.

## When to use what

| Environment | Use | Avoid |
|-------------|-----|--------|
| **Local desktop** | Interactive viewer, short smoke | Overnight 100 mm beads |
| **Colab / notebook** | Editable job cells, Drive sync, moderate beads | Full `validation.run_all` as primary CI |
| **HPC (SLURM)** | Full validation, long beads, reproducible batch | GGUI viewer (no display) |

## One-time setup

```bash
cd $SCRATCH   # prefer scratch over home for VTK outputs
git clone <repo-url> waam_twin
cd waam_twin   # directory that contains pyproject.toml

module load python/3.11 cuda/12.x   # names vary by site
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Prove CUDA before submitting jobs

```bash
nvidia-smi
python -c "import taichi as ti; ti.init(arch=ti.cuda); print(ti.cfg.arch)"
```

Silent CPU fallback wastes queue time. Fix Taichi/CUDA mismatch first.

If you skip `pip install -e .`, put the **parent** of the `waam_twin` package on
`PYTHONPATH` (see README Installation).

## SLURM scripts

Shipped under [`scripts/hpc/`](../scripts/hpc/):

| Script | Purpose | Typical walltime |
|--------|---------|------------------|
| `val_core.slurm` | Core CI validation | ~15–45 min |
| `val_full.slurm` | `WAAM_FULL_VALIDATION=1` | ~1–4 h |
| `run_production.slurm` | Headless bead + research bundle | job-dependent |
| `run_batch.py` | CLI used by production SLURM job | — |

Submit **from the repo root**:

```bash
mkdir -p logs runs
sbatch scripts/hpc/val_core.slurm
# after core passes:
sbatch scripts/hpc/val_full.slurm
# optional bead macrograph gates:
sbatch --export=ALL,WAAM_BEAD_VALIDATION=1 scripts/hpc/val_full.slurm

# production weld:
sbatch --export=ALL,WAAM_JOB=jobs/examples/bead_calibrate.yaml,WAAM_PRESET=high \
  scripts/hpc/run_production.slurm
```

Edit `#SBATCH` resource lines and `module load` comments for your site.
PBS/LSF sites: keep the shell body; replace `#SBATCH` with the local directives.

## Validation tiers

`run_all` defaults `WAAM_PRESET=minimal` — leave it. Tests own their grids;
raising to `high`/`ultra` mostly burns VRAM without changing gate logic.

```bash
# Core
WAAM_BACKEND=cuda WAAM_PRESET=minimal python -m waam_twin.validation.run_all

# Full (process + soak + held-out + intensive)
WAAM_FULL_VALIDATION=1 WAAM_BACKEND=cuda WAAM_PRESET=minimal \
  python -m waam_twin.validation.run_all

# Full + bead macrograph
WAAM_FULL_VALIDATION=1 WAAM_BEAD_VALIDATION=1 WAAM_BACKEND=cuda WAAM_PRESET=minimal \
  python -m waam_twin.validation.run_all
```

Re-run a single failure:

```bash
WAAM_BACKEND=cuda python -m waam_twin.validation.test_calibrated_pool
```

## Production batch (no notebook `N_STEPS`)

Physical timestep `dt` is fixed by LBM scaling (`dt = 0.1 · dx`). Torch advance
uses real travel speed. Step count must cover path length:

```text
steps_needed ≈ path_length_m / (travel_speed_m_s · dt)
```

`run_batch.py --n-steps auto` (and `run_path(..., n_steps=None)`) sizes this for
you. A fixed notebook-style `N_STEPS=12000` can end mid-bead on long paths.

```bash
python scripts/hpc/run_batch.py \
  --job jobs/examples/bead_calibrate.yaml \
  --preset high \
  --n-steps auto \
  --out runs/calibrate_high
```

## Grid / VRAM sizing

Two layers:

| Layer | Owns |
|-------|------|
| Job YAML | `domain_mm`, plate, process, `dx_mm` request |
| Preset | `vram_budget_mb`, `max_cells`, `target_dx_mm` |

`auto_grid` keeps domain fixed and **coarsens dx** until the optimistic VRAM
estimate and `max_cells` fit. The estimator undercounts full-physics fields
(Lorentz, VOF, tables, viewer buffers), so `ultra` on a true 16 GB card can
still OOM. Prefer:

- trim `domain_mm` or enable `enable_moving_window`
- use `high` with a realistic `dx` (~0.15–0.25 mm) before `ultra`
- leave ~15% VRAM headroom (`vram_budget_mb` × 0.85 is already applied)

See [HARDWARE.md](HARDWARE.md) and [VTK_EXPORT.md](VTK_EXPORT.md).

## Outputs

| Path | Contents |
|------|----------|
| `logs/*-%j.out` | SLURM stdout |
| `runs/<tag>/telemetry.json` | End-of-run telemetry |
| `runs/<tag>/bundle/` | Research VTK + meta (unless `WAAM_HEADLESS=1`) |

Keep large VTK trees on scratch; they are gitignored.

## Notebook on HPC?

Possible (Jupyter on a GPU node) but not recommended as the primary validation
or overnight path. Prefer SLURM + `run_batch.py` for queue accounting, logs, and
restarts. Use the notebook on Colab for interactive job editing and Drive sync.
