#!/bin/bash
#SBATCH --job-name=waam-val-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Full validation: process gates + soak + held-out sims + intensive physics.
# Submit from the waam_twin repo root after val_core has passed:
#   sbatch scripts/hpc/val_full.slurm
#
# Optional bead macrograph gates:
#   sbatch --export=ALL,WAAM_BEAD_VALIDATION=1 scripts/hpc/val_full.slurm

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
export WAAM_PRESET="${WAAM_PRESET:-minimal}"
export WAAM_FULL_VALIDATION=1
# export WAAM_BEAD_VALIDATION=1   # uncomment or pass via --export

echo "[slurm] host=$(hostname)  FULL_VALIDATION=1  BEAD=${WAAM_BEAD_VALIDATION:-0}"
nvidia-smi || true
python -c "import taichi as ti; ti.init(arch=ti.cuda); print('taichi arch=', ti.cfg.arch)"

python -m waam_twin.validation.run_all
echo "[slurm] val_full finished OK"
