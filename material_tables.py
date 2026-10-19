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


def sensible_knot_enthalpies(knots: list[tuple[float, float]]) -> list[float]:
    """Integral of cp from absolute zero, evaluated at each knot [J/kg].

    cp is held at the first knot from 0 K up to that knot, then piecewise
    linear. A constant table therefore stores cp·T, the same zero the
    historical rho·cp·T anchor used.
    """
    if not knots:
        return []
    h = [float(knots[0][1]) * float(knots[0][0])]
    for (t0, c0), (t1, c1) in zip(knots, knots[1:]):
        h.append(h[-1] + 0.5 * (float(c0) + float(c1)) * (float(t1) - float(t0)))
    return h


def sensible_enthalpy_J_kg(
    T: float,
    knots: list[tuple[float, float]],
    fallback: float,
) -> float:
    """Integral of cp from 0 K to T [J/kg]. Latent heat is not included."""
    if T <= 0.0:
        return 0.0
    if not knots:
        return float(fallback) * float(T)
    ts = [float(t) for t, _ in knots]
    cs = [float(c) for _, c in knots]
    hs = sensible_knot_enthalpies(knots)
    if T <= ts[0]:
        return cs[0] * float(T)
    if T >= ts[-1]:
        return hs[-1] + cs[-1] * (float(T) - ts[-1])
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        if t0 <= T <= t1:
            u = float(T) - t0
            dt = t1 - t0
            c0, c1 = cs[i], cs[i + 1]
            acc = hs[i] + c0 * u
            if dt > 1e-12:
                acc += 0.5 * (c1 - c0) / dt * u * u
            return acc
    return hs[-1]


def volumetric_enthalpy_J_m3(
    rho: float,
    cp_fallback: float,
    knots: list[tuple[float, float]],
    T: float,
    T_solidus: float,
    T_liquidus: float,
    L_fusion: float,
) -> float:
    """Volumetric enthalpy [J/m³]: rho·∫cp plus latent heat across the mush.

    Below the solidus this is the sensible integral. The full rho·L is in
    by the liquidus, spread linearly in temperature through the mush. With
    no knots and a constant cp it is rho·cp·T plus that jump, matching the
    old anchor.
    """
    sensible = float(rho) * sensible_enthalpy_J_kg(T, knots, cp_fallback)
    if float(T) <= float(T_solidus):
        return sensible
    latent = float(rho) * float(L_fusion)
    if float(T) >= float(T_liquidus):
        return sensible + latent
    span = float(T_liquidus) - float(T_solidus)
    if span <= 1e-6:
        return sensible + latent
    return sensible + latent * (float(T) - float(T_solidus)) / span
