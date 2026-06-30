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


@dataclass
class Gaussian2D:
    """Default surface Gaussian — production path."""

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi,
            arc_i, arc_j, arc_k,
            twin.Q_w, twin.sigma_cells,
            g.dt, g.dx ** 3, twin.eta,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


@dataclass
class Goldak3D:
    """Goldak double-ellipsoid heat source."""

    ff: float = 0.6
    fr: float = 0.4
    depth_front: float = 2.0
    depth_rear: float = 4.0
    sigma_scale: float = 1.0

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        sign = 1.0 if twin.travel_speed_m_s >= 0.0 else -1.0
        kernels.inject_goldak_heat(
            g.H, g.flags, g.phi,
            arc_i, arc_j, arc_k,
            twin.Q_w, twin.sigma_cells * self.sigma_scale,
            g.dt, g.dx ** 3, twin.eta,
            sign, self.ff, self.fr,
            self.depth_front, self.depth_rear,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


@dataclass
class ConicalVolume:
    """Conical volume — widened Gaussian until full cone geometry lands."""

    sigma_scale: float = 1.2

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi,
            arc_i, arc_j, arc_k - 1.5,
            twin.Q_w, twin.sigma_cells * self.sigma_scale,
            g.dt, g.dx ** 3, twin.eta,
            g.FLAG_SOLID, g.FLAG_GAS,
        )


def create_heat_source(name: str) -> ArcHeatSource:
    key = (name or "gaussian2d").lower().replace("-", "").replace("_", "")
    if key in ("goldak", "goldak3d", "doubleellipsoid"):
        return Goldak3D()
    if key in ("conical", "conicalvolume", "cone"):
        return ConicalVolume()
    return Gaussian2D()


__all__ = [
    "ArcHeatSource",
    "Gaussian2D",
    "Goldak3D",
    "ConicalVolume",
    "create_heat_source",
]
