# waam_twin — GPU environment image (Docker-capable HPC / workstations)
#
# The image installs Python, Quadrants, and this package. It does not start
# a job. You get a shell and run the same commands you would on the host.
#
# Build (from this directory — the repo root with pyproject.toml):
#   docker build -t waam-twin:latest .
#
# Open a shell on your working tree. -u keeps files you write owned by you.
#   docker run -it --rm --gpus all \
#     -u "$(id -u):$(id -g)" \
#     -e WAAM_BACKEND=cuda \
#     -v "$PWD:/app" \
#     waam-twin:latest
#
# Inside that shell, for example:
#   python scripts/hpc/run_batch.py \
#     --job jobs/examples/bead_on_plate.yaml \
#     --n-steps auto \
#     --out runs/bead_on_plate
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

# Shell, same as a login on this machine. Pass a command after the image
# name only when you want a one-off run instead of an interactive session.
CMD ["bash"]
