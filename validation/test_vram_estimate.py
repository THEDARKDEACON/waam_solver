"""
test_vram_estimate.py — auto_grid VRAM helper matches WAAMGrid.estimated_vram_mb.
"""

from __future__ import annotations

import sys

from waam_twin.grid import WAAMGrid
from waam_twin.materials import load_material
from waam_twin.runtime import (
    estimate_grid_vram_mb,
    init_taichi,
    resolve_grid,
    vram_flags_from_job,
)


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def run() -> None:
    init_taichi(backend="cpu")
    mat = load_material("materials/placeholders/ER70S-6.yaml")
    cases = (
        dict(lorentz=False, vof=False, export=False),
        dict(lorentz=True, vof=True, export=True),
        dict(lorentz=True, vof=False, export=True),
    )
    nx, ny, nz, tracers = 24, 16, 16, 128
    for flags in cases:
        grid = WAAMGrid(
            nx, ny, nz, 5e-4, mat, max_tracers=tracers,
            allocate_lorentz=flags["lorentz"],
            allocate_vof=flags["vof"],
            allocate_export=flags["export"],
        )
        est = estimate_grid_vram_mb(nx, ny, nz, tracers, **flags)
        got = grid.estimated_vram_mb()
        err = _rel_err(est, got)
        if err > 0.05:
            raise AssertionError(
                f"VRAM estimate {est:.4f} MB vs grid {got:.4f} MB "
                f"(rel {err:.3%}) flags={flags}"
            )
        print(f"[vram_estimate] flags={flags}  {got:.3f} MB  rel_err={err:.4%}")

    full = vram_flags_from_job({"simulation": {"physics_tier": "full"}})
    if full != (True, True, True):
        raise AssertionError(f"full tier should size full optional set, got {full}")
    thermal = vram_flags_from_job({"simulation": {"physics_tier": "thermal"}})
    if thermal[0] or thermal[1]:
        raise AssertionError(f"thermal tier should not assume Lorentz/VOF, got {thermal}")
    print("[vram_estimate] job flag peek OK")

    nx, ny, nz, dx = resolve_grid(
        (40.0, 40.0, 10.0), 0.25, 100_000,
        max_tracers=100, max_cells=200_000,
        auto_grid_enabled=False, lorentz=False, vof=False, export=False,
    )
    if abs(dx - 2.5e-4) > 1e-9 or (nx, ny, nz) != (160, 160, 40):
        raise AssertionError(
            f"auto_grid off must keep dx=0.25 mm, got {nx}×{ny}×{nz} dx={dx}"
        )
    nx2, ny2, nz2, dx2 = resolve_grid(
        (40.0, 40.0, 10.0), 0.25, 100_000,
        max_tracers=100, max_cells=200_000,
        auto_grid_enabled=True, lorentz=False, vof=False, export=False,
    )
    if not (dx2 > dx and nx2 * ny2 * nz2 <= 200_000):
        raise AssertionError(
            f"auto_grid on should coarsen under max_cells, got {nx2}×{ny2}×{nz2} dx={dx2}"
        )
    print(
        f"[vram_estimate] auto_grid off dx={dx*1e3:.3f} mm  "
        f"on dx={dx2*1e3:.3f} mm"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
