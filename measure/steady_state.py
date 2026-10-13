"""
Quasi-steady-state detection for bead runs.

Steady when W, D and peak T change by less than ``tol`` (default 1 %)
over one pool-length of travel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..twin import WAAMTwin


@dataclass
class SteadyStateDetector:
    tol: float = 0.01
    min_samples: int = 3
    history: list[dict] = field(default_factory=list)

    def observe(
        self,
        twin: "WAAMTwin",
        width_mm: float,
        depth_mm: float,
        peak_T_K: float,
        travel_mm: float,
    ) -> None:
        self.history.append(
            {
                "step": int(getattr(twin, "_step_n", 0)),
                "travel_mm": float(travel_mm),
                "W": float(width_mm),
                "D": float(depth_mm),
                "T": float(peak_T_K),
            }
        )

    def is_steady(self, pool_length_mm: float) -> bool:
        if len(self.history) < self.min_samples:
            return False
        L = max(float(pool_length_mm), 1.0)
        last = self.history[-1]
        # find oldest sample within one pool length of travel behind last
        target = last["travel_mm"] - L
        ref = None
        for h in reversed(self.history[:-1]):
            if h["travel_mm"] <= target:
                ref = h
                break
        if ref is None:
            return False

        def rel(a: float, b: float) -> float:
            return abs(a - b) / max(abs(b), 1e-9)

        return (
            rel(last["W"], ref["W"]) <= self.tol
            and rel(last["D"], ref["D"]) <= self.tol
            and rel(last["T"], ref["T"]) <= self.tol
        )


def run_to_steady_state(
    twin: "WAAMTwin",
    step_fn: Callable[[], None],
    *,
    max_steps: int,
    sample_every: int = 200,
    tol: float = 0.01,
    travel_speed_m_s: float | None = None,
) -> dict:
    """
    Advance ``step_fn`` until quasi-steady or ``max_steps``.

    ``step_fn`` should advance the twin by one physics step (and move the torch).
    Returns a dict with final metrics and whether steady was reached.
    """
    from .geometry import measure_fusion_zone_mm, measure_pool_length_mm

    det = SteadyStateDetector(tol=tol)
    v = float(travel_speed_m_s if travel_speed_m_s is not None else twin.travel_speed_m_s)
    dt = float(twin.grid.dt)
    steady = False
    for s in range(max_steps):
        step_fn()
        if (s + 1) % sample_every != 0 and s + 1 < max_steps:
            continue
        fz = measure_fusion_zone_mm(twin)
        L = measure_pool_length_mm(twin)
        travel_mm = (s + 1) * dt * v * 1000.0
        tel = twin.get_telemetry()
        peak = float(tel.get("peak_temp_K", 0.0))
        det.observe(twin, fz["pool_width_mm"], fz["pool_depth_mm"], peak, travel_mm)
        if det.is_steady(L):
            steady = True
            break

    fz = measure_fusion_zone_mm(twin)
    return {
        "steady": steady,
        "steps": int(getattr(twin, "_step_n", 0)),
        "travel_mm": float(getattr(twin, "_step_n", 0)) * dt * v * 1000.0,
        **fz,
        "pool_length_mm": measure_pool_length_mm(twin),
        "history": det.history,
    }
