"""Arc heat source models (waam_twin v2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .. import kernels

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


@dataclass
class Gaussian2D:
    """Default surface Gaussian — production path."""

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        pen_cells, enable = _arc_weight_args(twin)
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi, g.f_l,
            arc_i, arc_j, arc_k,
            twin.Q_w, twin.sigma_cells,
            g.dt, g.dx ** 3, twin.eta,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


@dataclass
class Goldak3D:
    """Goldak double-ellipsoid heat source (Goldak 1984)."""

    ff: float = 0.6
    fr: float = 0.4
    depth_front: float = 2.0
    depth_rear: float = 4.0
    sigma_scale: float = 1.0

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        sign = 1.0 if twin.travel_speed_m_s >= 0.0 else -1.0
        pen_cells, enable = _arc_weight_args(twin)
        kernels.inject_goldak_heat(
            g.H, g.flags, g.phi, g.f_l,
            arc_i, arc_j, arc_k,
            twin.Q_w, twin.sigma_cells * self.sigma_scale,
            g.dt, g.dx ** 3, twin.eta,
            sign, self.ff, self.fr,
            self.depth_front, self.depth_rear,
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
            g.H, g.flags, g.phi, g.f_l,
            arc_i, arc_j, arc_k - 1.5,
            twin.Q_w, twin.sigma_cells * self.sigma_scale,
            g.dt, g.dx ** 3, twin.eta,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


def create_heat_source(name: str, **kwargs) -> ArcHeatSource:
    key = (name or "gaussian2d").lower().replace("-", "").replace("_", "")
    if key in ("goldak", "goldak3d", "doubleellipsoid"):
        return Goldak3D(**kwargs)
    if key in ("conical", "conicalvolume", "cone"):
        return ConicalVolume(**kwargs)
    return Gaussian2D()


def goldak_from_job_mm(twin: "WAAMTwin", goldak_cfg: dict) -> Goldak3D:
    """Build Goldak3D with semi-axes in cells from job mm fields."""
    dx = twin.grid.dx
    depth_front = float(goldak_cfg.get("depth_front_mm", 0.9)) / (dx * 1000.0)
    depth_rear = float(goldak_cfg.get("depth_rear_mm", 1.8)) / (dx * 1000.0)
    return Goldak3D(
        ff=float(goldak_cfg.get("ff", 0.6)),
        fr=float(goldak_cfg.get("fr", 0.4)),
        depth_front=max(0.5, depth_front),
        depth_rear=max(0.5, depth_rear),
        sigma_scale=float(goldak_cfg.get("sigma_scale", 1.0)),
    )


__all__ = [
    "ArcHeatSource",
    "Gaussian2D",
    "Goldak3D",
    "ConicalVolume",
    "create_heat_source",
    "goldak_from_job_mm",
]
