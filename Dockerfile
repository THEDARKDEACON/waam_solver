# waam_twin — GPU headless batch image (Docker-capable HPC / workstations)
#
# Build (from this directory — the repo root with pyproject.toml):
#   docker build -t waam-twin:latest .
#
# Run (mount outputs so they survive the container):
#   mkdir -p runs
#   docker run --rm --gpus all \
#     -e WAAM_BACKEND=cuda \
#     -v "$PWD/runs:/app/runs" \
#     waam-twin:latest \
#     python scripts/hpc/run_batch.py \
#       --job jobs/examples/bead_on_plate_hires.yaml \
#       --n-steps auto \
#       --out runs/bead_on_plate_hires
#
# Match CUDA major version to the host driver (nvidia-smi). If Quadrants
# cannot init CUDA, try a newer/older nvidia/cuda tag or rebuild on the target node.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WAAM_BACKEND=cuda \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libgl1 \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy the package tree (see .dockerignore for exclusions), then install.
COPY . .

RUN pip3 install -U pip setuptools wheel \
    && pip3 install -e ".[export]" \
    && useradd --create-home --uid 1000 waam \
    && chown -R waam:waam /app

USER waam

# Default: higher-resolution bead. Override the command for other jobs.
CMD ["python", "scripts/hpc/run_batch.py", \
     "--job", "jobs/examples/bead_on_plate_hires.yaml", \
     "--n-steps", "auto", \
     "--out", "runs/bead_on_plate_hires"]
