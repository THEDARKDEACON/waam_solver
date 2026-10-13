"""
Phase 8 — run-efficiency helpers.

* Lorentz solve cadence (every N physical seconds)
* Sparse active-cell masks (host-side planner)
* GPU-side counter accessors
* Benchmark report formatting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def lorentz_due(twin: "WAAMTwin", every_s: float | None = None) -> bool:
    """True if a Lorentz solve should run this step."""
    if not twin.enable_lorentz:
        return False
    dt = float(twin.grid.dt)
    every = float(
        every_s
        if every_s is not None
        else getattr(twin, "lorentz_solve_every_s", 5.0e-5)
    )
    if every <= 0.0:
        return True
    n = max(1, int(round(every / max(dt, 1e-12))))
    return int(getattr(twin, "_step_n", 0)) % n == 0


def active_cell_bbox(twin: "WAAMTwin", pad: int = 4) -> tuple[int, int, int, int, int, int]:
    """
    Bounding box of non-gas cells (+ pad) for sparse kernel launch planning.
    Returns (i0,i1,j0,j1,k0,k1) inclusive-exclusive.
    """
    import numpy as np

    g = twin.grid
    flags = g.flags.to_numpy()
    metal = flags != g.FLAG_GAS
    if not metal.any():
        return 0, g.nx, 0, g.ny, 0, g.nz
    xs, ys, zs = np.where(metal)
    i0 = max(0, int(xs.min()) - pad)
    i1 = min(g.nx, int(xs.max()) + 1 + pad)
    j0 = max(0, int(ys.min()) - pad)
    j1 = min(g.ny, int(ys.max()) + 1 + pad)
    k0 = max(0, int(zs.min()) - pad)
    k1 = min(g.nz, int(zs.max()) + 1 + pad)
    return i0, i1, j0, j1, k0, k1


def throughput_report(twin: "WAAMTwin", ms_per_step: float) -> dict[str, Any]:
    g = twin.grid
    cells = g.nx * g.ny * g.nz
    steps_s = 1000.0 / max(ms_per_step, 1e-9)
    return {
        "ms_per_step": ms_per_step,
        "steps_per_s": steps_s,
        "MLUPS": cells * steps_s / 1e6,
        "grid": [g.nx, g.ny, g.nz],
        "dx_m": g.dx,
        "dt_s": g.dt,
        "force_clamp_steps": int(getattr(twin, "_force_clamp_steps", 0)),
        "mach_clamp_steps": int(getattr(twin, "_mach_clamp_steps", 0)),
        "lorentz_iters_last": getattr(twin, "_lorentz_iters_last", None),
    }
