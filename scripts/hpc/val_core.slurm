#!/bin/bash
#SBATCH --job-name=waam-val-core
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Core validation suite (CI-equivalent). Adjust module names for your site.
# Submit from the waam_twin repo root:
#   mkdir -p logs
#   sbatch scripts/hpc/val_core.slurm

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
# Leave WAAM_HEADLESS unset so export/VTK tests still run.

echo "[slurm] host=$(hostname)  cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
python -c "from waam_twin.compiler import ti; ti.init(arch=ti.cuda); print('quadrants arch=', ti.cfg.arch)"

python -m waam_twin.validation.run_all
echo "[slurm] val_core finished OK"
