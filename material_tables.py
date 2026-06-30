"""Piecewise-linear material property tables (waam_twin v2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialTables:
    """Temperature-dependent properties as [T_K, value] knot lists."""

    cp: list[tuple[float, float]] = field(default_factory=list)
    k: list[tuple[float, float]] = field(default_factory=list)
    mu: list[tuple[float, float]] = field(default_factory=list)
    dgamma_dT: list[tuple[float, float]] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(self.cp or self.k or self.mu or self.dgamma_dT)


def _parse_knots(raw: list) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for row in raw or []:
        if len(row) != 2:
            continue
        out.append((float(row[0]), float(row[1])))
    out.sort(key=lambda p: p[0])
    return out


def tables_from_yaml(data: dict | None) -> MaterialTables:
    if not data:
        return MaterialTables()
    return MaterialTables(
        cp=_parse_knots(data.get("cp", [])),
        k=_parse_knots(data.get("k", [])),
        mu=_parse_knots(data.get("mu", [])),
        dgamma_dT=_parse_knots(data.get("dgamma_dT", [])),
    )


def interp_linear(T: float, knots: list[tuple[float, float]], fallback: float) -> float:
    if not knots:
        return fallback
    if T <= knots[0][0]:
        return knots[0][1]
    if T >= knots[-1][0]:
        return knots[-1][1]
    for (t0, v0), (t1, v1) in zip(knots[:-1], knots[1:]):
        if t0 <= T <= t1:
            if t1 == t0:
                return v0
            w = (T - t0) / (t1 - t0)
            return v0 + w * (v1 - v0)
    return fallback
