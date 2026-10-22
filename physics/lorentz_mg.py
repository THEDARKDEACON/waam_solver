"""
Multigrid V-cycle scaffolding for the electric potential (Phase 3 / H5).

Uses residual-based convergence (‖r‖₂ / ‖b‖₂ < tol) rather than per-iteration
Δφ. Falls back to the existing Jacobi loop when coarse grids are unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logging_util as log

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid


def residual_converged(g: "WAAMGrid", tol: float) -> tuple[bool, float]:
    """Return (ok, ||b - A phi|| / ||b||) from the last Lorentz solve."""
    rel = float(getattr(g, "_elec_rel_residual", float("inf")))
    return rel < tol, rel


def solve_lorentz_multigrid(
    twin: "WAAMTwin",
    g: "WAAMGrid",
    arc_i: float,
    arc_j: float,
    arc_k: float,
) -> bool:
    """
    Attempt a residual-converged Lorentz solve.

    Currently wraps the Jacobi solve with a stricter residual check and
    optional re-solve when the first pass fails. Full geometric MG (restrict /
    prolong) is staged for HPC; the API is stable for callers.
    """
    from . import weld_forces

    weld_forces.solve_lorentz(twin, g, arc_i, arc_j, arc_k)
    ok, rel = residual_converged(g, float(getattr(twin, "lorentz_jacobi_tol", 1e-6)))
    if ok:
        return True
    # One retry with doubled iteration budget
    old = int(twin.lorentz_jacobi_iters)
    twin.lorentz_jacobi_iters = min(old * 2, 2000)
    twin._lorentz_warm = True
    try:
        weld_forces.solve_lorentz(twin, g, arc_i, arc_j, arc_k)
    finally:
        twin.lorentz_jacobi_iters = old
    ok, rel = residual_converged(g, float(getattr(twin, "lorentz_jacobi_tol", 1e-6)))
    if not ok:
        log.warning(
            f"[lorentz_mg] residual not converged after retry (rel={rel:.2e})"
        )
    return ok
