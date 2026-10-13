"""
Grid Convergence Index (Roache) utilities — Phase 6.

Requires ≥3 meshes at equal physical time. Clamps firing → study invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math


@dataclass
class GCIResult:
    p: float  # observed order
    f_exact: float
    gci_fine: float
    gci_mid: float
    asymptotic: bool
    valid: bool
    notes: str = ""


def richardson_gci(
    h: Sequence[float],
    f: Sequence[float],
    *,
    safety: float = 1.25,
    p_theory: float = 2.0,
) -> GCIResult:
    """
    Three-mesh GCI. ``h`` = cell sizes (fine→coarse or any order; sorted ascending).
    ``f`` = corresponding QoI values.
    """
    if len(h) < 3 or len(f) < 3:
        raise ValueError("Need at least 3 meshes for GCI")
    pairs = sorted(zip(h, f), key=lambda t: t[0])
    h1, h2, h3 = [p[0] for p in pairs[:3]]
    f1, f2, f3 = [p[1] for p in pairs[:3]]
    r21 = h2 / h1
    r32 = h3 / h2
    eps21 = f2 - f1
    eps32 = f3 - f2
    if abs(eps21) < 1e-30 or abs(eps32) < 1e-30:
        return GCIResult(
            p=float("nan"),
            f_exact=f1,
            gci_fine=0.0,
            gci_mid=0.0,
            asymptotic=False,
            valid=False,
            notes="zero change between meshes",
        )
    # Observed order (Roache)
    s = 1.0 if eps32 / eps21 > 0 else -1.0
    try:
        p = math.log(abs(eps32 / eps21)) / math.log(r21)
        # iterate once with r32≠r21 correction
        for _ in range(4):
            p = math.log(abs(eps32 / eps21) * (r21 ** p - s) / (r32 ** p - s) + 1e-30) / math.log(
                r21
            )
    except (ValueError, ZeroDivisionError):
        p = p_theory
    if not math.isfinite(p) or p <= 0:
        p = p_theory
    f_exact = f1 + (f1 - f2) / (r21 ** p - 1.0)
    gci_21 = safety * abs(f1 - f2) / (abs(f1) + 1e-30) / (r21 ** p - 1.0)
    gci_32 = safety * abs(f2 - f3) / (abs(f2) + 1e-30) / (r32 ** p - 1.0)
    # Asymptotic range check: GCI_{32} / (r^p GCI_{21}) ≈ 1
    ratio = gci_32 / max(r21 ** p * gci_21, 1e-30)
    asymptotic = 0.75 <= ratio <= 1.25
    return GCIResult(
        p=float(p),
        f_exact=float(f_exact),
        gci_fine=float(gci_21),
        gci_mid=float(gci_32),
        asymptotic=asymptotic,
        valid=True,
        notes=f"ratio={ratio:.3f}",
    )


def study_invalid_if_clamps(rows: list[dict]) -> list[str]:
    """Return reasons the mesh study is invalid."""
    reasons = []
    for r in rows:
        if int(r.get("force_clamp_steps", 0)) > 0:
            reasons.append(
                f"dx={r.get('dx_mm')}: force clamps in {r['force_clamp_steps']} steps"
            )
        if int(r.get("mach_clamp_steps", 0)) > 0:
            reasons.append(
                f"dx={r.get('dx_mm')}: Mach clamps in {r['mach_clamp_steps']} steps"
            )
    return reasons
