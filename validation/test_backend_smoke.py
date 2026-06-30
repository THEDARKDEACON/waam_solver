"""
test_backend_smoke.py — Init + thermal diffusion smoke on available backends.

CPU is required; Vulkan/CUDA attempted when WAAM_BACKEND_MATRIX=1.
"""

from __future__ import annotations

import os
import sys


def _smoke_backend(backend: str) -> None:
    from waam_twin.platform import init_taichi
    from waam_twin.validation import test_thermal_diffusion

    os.environ["WAAM_BACKEND"] = backend
    init_taichi(backend=backend)
    test_thermal_diffusion.run(n_steps=20, threshold=20.0)


def run() -> None:
    _smoke_backend("cpu")
    print("[backend_smoke] cpu OK")

    if os.environ.get("WAAM_BACKEND_MATRIX") == "1":
        for backend in ("vulkan", "cuda"):
            try:
                _smoke_backend(backend)
                print(f"[backend_smoke] {backend} OK")
            except Exception as exc:
                print(f"[backend_smoke] {backend} skip: {exc}")
    else:
        print("[backend_smoke] optional vulkan/cuda skipped (set WAAM_BACKEND_MATRIX=1)")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
