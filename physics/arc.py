"""Arc heat source models (waam_twin v2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .. import kernels
from .. import logging_util as log

if TYPE_CHECKING:
    from ..grid import WAAMGrid
    from ..twin import WAAMTwin


class ArcHeatSource(Protocol):
    def inject(
        self,
        twin: "WAAMTwin",
        g: "WAAMGrid",
        arc_i: float,
        arc_j: float,
        arc_k: float,
    ) -> None: ...


def _arc_weight_args(twin: "WAAMTwin") -> tuple[float, int]:
    pen_cells = twin.arc_penetration_m / twin.grid.dx
    enable = 1 if twin.arc_surface_weighting else 0
    return pen_cells, enable


def normalize_goldak_fractions(ff: float, fr: float) -> tuple[float, float]:
    """
    Enforce Goldak f_f + f_r = 2.

    Fractions that sum to ~1 (common YAML mistake) are scaled ×2.
    Other sums are renormalized to 2 with a warning.
    """
    s = float(ff) + float(fr)
    if s <= 1e-12:
        log.warning("[Goldak] ff+fr≤0; using defaults ff=0.6, fr=1.4")
        return 0.6, 1.4
    if abs(s - 2.0) <= 0.05:
        return float(ff), float(fr)
    if abs(s - 1.0) <= 0.05:
        log.warning(
            f"[Goldak] ff+fr={s:.3f} (unity fractions); scaling ×2 → "
            f"ff={2.0 * ff:.3f}, fr={2.0 * fr:.3f}"
        )
        return 2.0 * float(ff), 2.0 * float(fr)
    log.warning(
        f"[Goldak] ff+fr={s:.3f} ≠ 2; renormalizing to sum 2 "
        f"(ff={2.0 * ff / s:.3f}, fr={2.0 * fr / s:.3f})"
    )
    return 2.0 * float(ff) / s, 2.0 * float(fr) / s


@dataclass
class Gaussian2D:
    """Default surface Gaussian — production path."""

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        pen_cells, enable = _arc_weight_args(twin)
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi, g.f_l, g.arc_norm_buf,
            arc_i, arc_j, arc_k,
            twin.Q_w, twin.sigma_cells,
            g.dt, g.dx ** 3, twin.eta,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


@dataclass
class Goldak3D:
    """Goldak double-ellipsoid heat source (Goldak 1984).

    Semi-axes are in lattice cells. Values ≤0 are resolved at inject time
    from the twin's arc σ and legacy depth fields.
    """

    ff: float = 0.6
    fr: float = 1.4
    a_front: float = 0.0
    a_rear: float = 0.0
    b: float = 0.0
    c: float = 0.0
    # Legacy / fallback when a,b,c not set from job mm fields
    depth_front: float = 2.0
    depth_rear: float = 4.0
    sigma_scale: float = 1.0

    def _resolved_axes(self, twin: "WAAMTwin") -> tuple[float, float, float, float]:
        sig = max(0.5, twin.sigma_cells * self.sigma_scale)
        a_f = self.a_front if self.a_front > 0.0 else sig
        a_r = self.a_rear if self.a_rear > 0.0 else max(sig, 2.0 * sig)
        b = self.b if self.b > 0.0 else sig
        if self.c > 0.0:
            c = self.c
        else:
            # Prefer front depth; fall back to rear if only that was set.
            c = max(0.5, self.depth_front if self.depth_front > 0.0 else self.depth_rear)
        return a_f, a_r, b, c

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        sign = 1.0 if twin.travel_speed_m_s >= 0.0 else -1.0
        pen_cells, enable = _arc_weight_args(twin)
        a_f, a_r, b, c = self._resolved_axes(twin)
        kernels.inject_goldak_heat(
            g.H, g.flags, g.phi, g.f_l, g.arc_norm_buf,
            arc_i, arc_j, arc_k,
            twin.Q_w,
            g.dt, g.dx ** 3, twin.eta,
            sign, self.ff, self.fr,
            a_f, a_r, b, c,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


@dataclass
class ConicalVolume:
    """Conical volume — widened Gaussian until full cone geometry lands."""

    sigma_scale: float = 1.2

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        pen_cells, enable = _arc_weight_args(twin)
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi, g.f_l, g.arc_norm_buf,
            arc_i, arc_j, arc_k - 1.5,
            twin.Q_w, twin.sigma_cells * self.sigma_scale,
            g.dt, g.dx ** 3, twin.eta,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


def create_heat_source(name: str, **kwargs) -> ArcHeatSource:
    key = (name or "gaussian2d").lower().replace("-", "").replace("_", "")
    if key in ("goldak", "goldak3d", "doubleellipsoid"):
        ff, fr = normalize_goldak_fractions(
            float(kwargs.pop("ff", 0.6)),
            float(kwargs.pop("fr", 1.4)),
        )
        return Goldak3D(ff=ff, fr=fr, **kwargs)
    if key in ("conical", "conicalvolume", "cone"):
        return ConicalVolume(**kwargs)
    return Gaussian2D()


def goldak_from_job_mm(twin: "WAAMTwin", goldak_cfg: dict) -> Goldak3D:
    """Build Goldak3D with semi-axes in cells from job mm fields.

    Preferred keys (PHYSICS_FORCE_CORRECTNESS_SPEC §5.4):
      a_front_mm, a_rear_mm, b_mm, c_mm, ff, fr  (ff+fr → 2)

    Legacy keys still accepted:
      depth_front_mm / depth_rear_mm → c (and depth_* cell fallbacks)
      sigma_scale → scales a,b from arc σ when a_*/b not given
    """
    dx = twin.grid.dx
    dx_mm = dx * 1000.0
    cfg = goldak_cfg or {}

    ff, fr = normalize_goldak_fractions(
        float(cfg.get("ff", 0.6)),
        float(cfg.get("fr", 1.4)),
    )
    sigma_scale = float(cfg.get("sigma_scale", 1.0))
    sig_cells = max(0.5, twin.sigma_cells * sigma_scale)

    def _mm_to_cells(key: str, default_cells: float | None = None) -> float:
        if key in cfg and cfg[key] is not None:
            return max(0.5, float(cfg[key]) / dx_mm)
        if default_cells is None:
            return 0.0
        return max(0.5, float(default_cells))

    # Explicit Goldak axes (0 ⇒ resolve at inject from σ / legacy depths)
    a_front = _mm_to_cells("a_front_mm", None)
    a_rear = _mm_to_cells("a_rear_mm", None)
    b = _mm_to_cells("b_mm", None)
    c = _mm_to_cells("c_mm", None)

    depth_front = _mm_to_cells("depth_front_mm", 2.0)
    depth_rear = _mm_to_cells("depth_rear_mm", 4.0)

    # If only legacy depths given, map: a~σ, b~σ, c~depth_front
    if a_front <= 0.0 and a_rear <= 0.0 and b <= 0.0 and c <= 0.0:
        a_front = sig_cells
        a_rear = max(sig_cells, 2.0 * sig_cells)
        b = sig_cells
        c = depth_front
        log.info(
            f"[Goldak] legacy depths → a_f={a_front:.2f} a_r={a_rear:.2f} "
            f"b={b:.2f} c={c:.2f} cells (σ_scale={sigma_scale})"
        )
    elif c <= 0.0:
        c = depth_front

    return Goldak3D(
        ff=ff,
        fr=fr,
        a_front=a_front,
        a_rear=a_rear,
        b=b,
        c=c,
        depth_front=depth_front,
        depth_rear=depth_rear,
        sigma_scale=sigma_scale,
    )


__all__ = [
    "ArcHeatSource",
    "Gaussian2D",
    "Goldak3D",
    "ConicalVolume",
    "create_heat_source",
    "goldak_from_job_mm",
    "normalize_goldak_fractions",
]
