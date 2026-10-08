# waam_twin — GPU image with the same Python + system deps as a local desktop
#
# Includes CUDA compute *and* OpenGL/X11 client libraries so
#   python -m waam_twin.viewer …
# works when the host has a display (local desktop, RDP/xrdp, X11).
# Headless batch still works the same way (no DISPLAY needed).
#
# The base image is Ubuntu so the container has its own Python and CUDA
# userland. The host can be RHEL 9.x; it does not need to match.
#
# ---------------------------------------------------------------------------
# Build (from this directory — pyproject.toml + Dockerfile):
#   podman build -t waam-twin:latest .
#   # or: docker build -t waam-twin:latest .
#
# Interactive shell + GPU (headless OK):
#   ./scripts/hpc/run_desktop.sh
#
# Viewer on RDP / local desktop (forwards $DISPLAY):
#   ./scripts/hpc/run_desktop.sh \
#     python -m waam_twin.viewer --job jobs/examples/bead_calibrate.yaml
#
# Batch (no GUI):
#   ./scripts/hpc/run_desktop.sh \
#     python scripts/hpc/run_batch.py \
#       --job jobs/examples/bead_on_plate_hires.yaml \
#       --n-steps auto --out runs/bead_on_plate_hires
#
# Manual podman equivalent (RHEL sites):
#   podman run -it --rm --device nvidia.com/gpu=all \
#     --userns=keep-id --group-add keep-groups \
#     --security-opt label=disable \
#     -u "$(id -u):$(id -g)" \
#     -e NVIDIA_VISIBLE_DEVICES=all \
#     -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     -e WAAM_BACKEND=cuda \
#     -e DISPLAY="$DISPLAY" \
#     -e QT_X11_NO_MITSHM=1 \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     -v "$PWD:/app:Z" -w /app \
#     waam-twin:latest bash
#
# Drop `:Z` on NFS if relabel fails. Match CUDA major to host `nvidia-smi`.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WAAM_BACKEND=cuda \
    PIP_NO_CACHE_DIR=1 \
    # VTK/PyVista: prefer system GL; OSMesa is a fallback for offscreen export
    PYVISTA_OFF_SCREEN=false \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        # Python toolchain
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        # OpenGL / EGL (PyVista + VTK wheels)
        libgl1 \
        libglib2.0-0 \
        libopengl0 \
        libegl1 \
        libgles2 \
        libglu1-mesa \
        libosmesa6 \
        mesa-utils \
        # X11 client (viewer window on host DISPLAY / RDP)
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxkbcommon0 \
        libxcb1 \
        libxcb-xinerama0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-cursor0 \
        libxi6 \
        libxrandr2 \
        libxxf86vm1 \
        libxcursor1 \
        libxinerama1 \
        libxfixes3 \
        # Fonts / text HUD
        libfontconfig1 \
        libfreetype6 \
        fonts-dejavu-core \
        # Runtime helpers
        libgomp1 \
        xvfb \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy the package tree (see .dockerignore for exclusions), then install.
COPY . .

# Same extras a local desktop uses for viewer + VTK export (+ sympy derive tools).
RUN pip3 install -U pip setuptools wheel \
    && pip3 install -e ".[export,derive]" \
    && useradd --create-home --uid 1000 waam \
    && chown -R waam:waam /app

USER waam

# Shell, same as a login on this machine. Pass a command after the image
# name only when you want a one-off run instead of an interactive session.
CMD ["bash"]
