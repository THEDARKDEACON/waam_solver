"""Stick-out resistance and CTWD coupling for wire preheat (WP5)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .deposition_balance import wire_mass_flux_kg_s

if TYPE_CHECKING:
    from ..twin import WAAMTwin
    from ..grid import WAAMGrid


def stickout_resistance_ohm(twin: "WAAMTwin") -> float:
    """R_stick = ρ_e L / A_wire."""
    L = max(twin.stickout_m, 1e-6)
    d = twin.wire_diameter_m
    area = math.pi * (d * 0.5) ** 2
    return twin.rho_e_ohm_m * L / max(area, 1e-12)


def droplet_entry_temperature_K(twin: "WAAMTwin") -> float:
    """Wire preheat from I²R stick-out added to liquidus + superheat."""
    base = twin.mat.T_liquidus + twin.deposition_superheat_K
    if not twin.enable_ctwd:
        return base
    I = twin.welding_current_A
    R = stickout_resistance_ohm(twin)
    P = twin.eta_stick * I * I * R
    mdot = wire_mass_flux_kg_s(twin)
    if mdot <= 0:
        return base
    dT = P / (mdot * twin.mat.cp)
    cap = twin.T_vapor_cap_K - 50.0
    return min(cap, base + dT)


def measure_bead_height_m(twin: "WAAMTwin", g: "WAAMGrid") -> float:
    """Max z of deposited SOLID above substrate reference nz_solid."""
    flags = g.flags.to_numpy()
    nz_ref = twin.nz_solid
    solid = flags == g.FLAG_SOLID
    if not solid.any():
        return 0.0
    z_idx = solid.nonzero()[2]
    dep = z_idx[z_idx >= nz_ref]
    if dep.size == 0:
        return 0.0
    return float((dep.max() - nz_ref + 1) * g.dx)


def update_ctwd(twin: "WAAMTwin", g: "WAAMGrid") -> None:
    """Open-loop CTWD from measured bead height vs nominal layer step."""
    if not twin.enable_ctwd:
        return
    h_meas = measure_bead_height_m(twin, g)
    twin._bead_height_m = h_meas
    twin.ctwd_m = max(twin.ctwd_nominal_m, twin.ctwd_nominal_m + (h_meas - twin.layer_height_m))
