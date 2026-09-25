# waam_twin — GPU environment image (Docker-capable HPC / workstations)
#
# The image installs Python, Quadrants, and this package. It does not start
# a job. You get a shell and run the same commands you would on the host.
#
# The base image is Ubuntu so the container has its own Python and CUDA
# userland. The host can be RHEL 9.8; it does not need to match.
#
# RHEL 9 ships Podman, not Docker. From this directory (pyproject.toml):
#   podman build -t waam-twin:latest .
#   podman run -it --rm --device nvidia.com/gpu=all \
#     --userns=keep-id \
#     --group-add keep-groups \
#     --security-opt label=disable \
#     -u "$(id -u):$(id -g)" \
#     -e NVIDIA_VISIBLE_DEVICES=all \
#     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#     -e WAAM_BACKEND=cuda \
#     -v "$PWD:/app:Z" \
#     -w /app \
#     waam-twin:latest \
#     bash
#
# If the site installed the podman-docker wrapper, `docker build` / `docker run`
# call Podman. `--gpus all` is that Docker-compatible spelling; Podman itself
# uses `--device nvidia.com/gpu=all`. Drop `:Z` on an NFS home directory if
# the relabel fails. A site that only allows Apptainer will not run this
# directly — build a SIF from the image instead.
#
# Docker on a machine that actually has the Docker daemon:
#   docker build -t waam-twin:latest .
#   docker run -it --rm --gpus all \
#     -u "$(id -u):$(id -g)" \
#     -e WAAM_BACKEND=cuda \
#     -v "$PWD:/app" \
#     waam-twin:latest \
#     bash
#
# `bash` is the whole command. Nothing runs until you type it.
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
