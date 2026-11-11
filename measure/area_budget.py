"""Where does the bead cross-section area come from?

``steady_cross_section`` reports ``area_mm2`` = metal fill above the original
plate face. If it exceeds the wire supply π/4·d²·v_wire/v_travel, the excess can be
1. mass gain   — the free-surface scheme creates metal (domain volume grows
                 faster than the deposition step books it);
2. booking     — the deposition step books more than the wire feeds;
3. lift-out    — plate metal pushed above the face, leaving a deficit below it
                 (the signed below-face change cancels it in ``area_net_mm2``);
4. humping     — the window sits on a local bulge (``area_profile``).

``snapshot`` is taken right after ``twin.reset()``; ``area_budget`` at the end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def _fill(twin: "WAAMTwin") -> np.ndarray:
    g = twin.grid
    metal = g.flags.to_numpy() != g.FLAG_GAS
    return np.clip(g.phi.to_numpy(), 0.0, 1.0) * metal


def snapshot(twin: "WAAMTwin") -> dict[str, Any]:
    fill = _fill(twin)
    return {"fill_below": fill[:, :, : int(twin.nz_solid)].copy(), "vol_cells": float(fill.sum())}


def area_budget(twin: "WAAMTwin", snap: dict[str, Any], x0_mm: float, x1_mm: float,
                profile_step_mm: float = 4.0) -> dict[str, Any]:
    from ..physics.deposition_balance import wire_mass_flux_kg_s

    g = twin.grid
    dx_mm = float(g.dx) * 1e3
    cell_mm2 = dx_mm * dx_mm
    cell_m3 = float(g.dx) ** 3
    nz = int(twin.nz_solid)
    fill = _fill(twin)

    i0 = max(0, min(g.nx - 1, int(round(x0_mm / dx_mm))))
    i1 = max(i0, min(g.nx - 1, int(round(x1_mm / dx_mm))))
    above = fill[i0:i1 + 1, :, nz:].sum(axis=(1, 2)) * cell_mm2
    below = (fill[i0:i1 + 1, :, :nz] - snap["fill_below"][i0:i1 + 1]).sum(axis=(1, 2)) * cell_mm2
    # interface share: area carried by partially filled cells above the face
    part = fill[i0:i1 + 1, :, nz:]
    iface = np.where((part > 0.0) & (part < 1.0), part, 0.0).sum(axis=(1, 2)) * cell_mm2

    rho = float(twin.mat.rho)
    t_weld = float(getattr(twin, "_welding_time_s", 0.0) or 0.0)
    v_fed = wire_mass_flux_kg_s(twin) * t_weld / rho
    v_booked = float(getattr(twin, "_deposited_volume_m3", 0.0))
    v_gain = (float(fill.sum()) - snap["vol_cells"]) * cell_m3

    # along-bead profile of the above-face area (humping check)
    n_step = max(1, int(round(profile_step_mm / dx_mm)))
    prof_all = fill[:, :, nz:].sum(axis=(1, 2)) * cell_mm2
    profile = [
        [round((i + n_step / 2) * dx_mm, 1), round(float(prof_all[i:i + n_step].mean()), 2)]
        for i in range(0, g.nx - n_step + 1, n_step)
    ]
    return {
        "area_above_mm2": float(above.mean()),
        "area_below_change_mm2": float(below.mean()),
        "area_net_mm2": float((above + below).mean()),
        "area_iface_mm2": float(iface.mean()),
        "vol_wire_fed_mm3": v_fed * 1e9,
        "vol_booked_mm3": v_booked * 1e9,
        "vol_gain_mm3": v_gain * 1e9,
        "booked_over_fed": v_booked / v_fed if v_fed > 0 else float("nan"),
        "gain_over_booked": v_gain / v_booked if v_booked > 0 else float("nan"),
        "area_profile": profile,
    }
