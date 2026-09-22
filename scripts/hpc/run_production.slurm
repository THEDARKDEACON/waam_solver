#!/bin/bash
#SBATCH --job-name=waam-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Production bead run (headless). Edit JOB / PRESET / OUT below or override:
#   sbatch --export=ALL,WAAM_JOB=jobs/examples/bead_on_plate.yaml,WAAM_PRESET=high \
#     scripts/hpc/run_production.slurm

set -euo pipefail

# --- site modules (edit) ---
# module load python/3.11
# module load cuda/12.2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs runs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export WAAM_BACKEND="${WAAM_BACKEND:-cuda}"
JOB="${WAAM_JOB:-jobs/examples/bead_calibrate.yaml}"
PRESET="${WAAM_PRESET:-high}"
OUT="${WAAM_OUT:-runs/prod_${SLURM_JOB_ID:-manual}}"

echo "[slurm] job=$JOB  preset=$PRESET  out=$OUT"
nvidia-smi || true
python -c "from waam_twin.compiler import ti; ti.init(arch=ti.cuda); print('quadrants arch=', ti.cfg.arch)"

# --n-steps auto covers the full torch path (do not under-size like notebook N_STEPS).
python scripts/hpc/run_batch.py \
  --job "$JOB" \
  --preset "$PRESET" \
  --n-steps auto \
  --out "$OUT"

echo "[slurm] production finished → $OUT"
