#!/usr/bin/env bash
# Run waam-twin container like a local desktop session (GPU + optional GUI).
#
# Usage (from waam_twin repo root — directory with Dockerfile):
#   ./scripts/hpc/run_desktop.sh
#   ./scripts/hpc/run_desktop.sh python -m waam_twin.viewer --job jobs/examples/bead_calibrate.yaml
#   ./scripts/hpc/run_desktop.sh python scripts/hpc/run_batch.py --job jobs/examples/bead_calibrate.yaml --n-steps auto --out runs/cal
#
# Env overrides:
#   WAAM_IMAGE=waam-twin:latest
#   WAAM_NO_X11=1          # skip DISPLAY / X11 mount (batch-only)
#   WAAM_CONTAINER=podman  # force engine (auto: podman if present, else docker)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${WAAM_IMAGE:-waam-twin:latest}"
ENGINE="${WAAM_CONTAINER:-}"

if [[ -z "$ENGINE" ]]; then
  if command -v podman >/dev/null 2>&1; then
    ENGINE=podman
  elif command -v docker >/dev/null 2>&1; then
    ENGINE=docker
  else
    echo "Neither podman nor docker found in PATH." >&2
    exit 1
  fi
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

# Allow local X clients from the container (RDP / desktop). Harmless if unused.
if [[ "${WAAM_NO_X11:-0}" != "1" ]] && command -v xhost >/dev/null 2>&1; then
  xhost +local: >/dev/null 2>&1 || true
fi

COMMON=(
  --rm -it
  -e WAAM_BACKEND=cuda
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
  -e QT_X11_NO_MITSHM=1
  -e PYVISTA_OFF_SCREEN="${PYVISTA_OFF_SCREEN:-false}"
  -w /app
)

if [[ "${WAAM_NO_X11:-0}" != "1" && -n "${DISPLAY:-}" ]]; then
  COMMON+=(-e "DISPLAY=${DISPLAY}")
  if [[ -d /tmp/.X11-unix ]]; then
    COMMON+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  fi
  # xrdp / some desktops also use ~/.Xauthority
  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    COMMON+=(-e "XAUTHORITY=/tmp/.Xauthority" -v "${XAUTHORITY}:/tmp/.Xauthority:ro")
  elif [[ -f "${HOME}/.Xauthority" ]]; then
    COMMON+=(-e "XAUTHORITY=/tmp/.Xauthority" -v "${HOME}/.Xauthority:/tmp/.Xauthority:ro")
  fi
fi

if [[ "$ENGINE" == "podman" ]]; then
  # Prefer :Z for SELinux; fall back without if the site uses NFS and relabel fails.
  VOL=(-v "${ROOT}:/app:Z")
  exec podman run \
    --device nvidia.com/gpu=all \
    --userns=keep-id \
    --group-add keep-groups \
    --security-opt label=disable \
    -u "$(id -u):$(id -g)" \
    "${COMMON[@]}" \
    "${VOL[@]}" \
    "$IMAGE" \
    "$@"
fi

exec docker run \
  --gpus all \
  -u "$(id -u):$(id -g)" \
  "${COMMON[@]}" \
  -v "${ROOT}:/app" \
  "$IMAGE" \
  "$@"
