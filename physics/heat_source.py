"""
Phase 4 — energy-split arc / droplet heat and surface-flux model.

η·V·I = Q_arc + Q_droplet with Q_droplet = ṁ · (h_drop − h_ambient).
Surface flux uses σ(I, V, CTWD) Gaussian.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def droplet_enthalpy_J_kg(twin: "WAAMTwin", T_drop_K: float) -> float:
    """Specific enthalpy of liquid droplet metal [J/kg]."""
    mat = twin.mat
    cp = float(mat.cp_at(0.5 * (mat.T_solidus + mat.T_liquidus)))
    # M7/M9-consistent: h = cp·T_sol + L + cp·(T − T_liq) for T ≥ T_liq
    return (
        cp * float(mat.T_solidus)
        + float(mat.L_fusion)
        + cp * (float(T_drop_K) - float(mat.T_liquidus))
    )


def ambient_enthalpy_J_kg(twin: "WAAMTwin") -> float:
    mat = twin.mat
    cp = float(mat.cp_at(twin.T_amb))
    return cp * float(twin.T_amb)


def wire_mass_rate_kg_s(twin: "WAAMTwin") -> float:
    """ṁ = ρ · (π/4) d² · v_feed."""
    d = float(twin.wire_diameter_m)
    return float(twin.mat.rho) * (math.pi / 4.0) * d * d * float(twin.wire_feed_m_s)


def energy_split_W(twin: "WAAMTwin", T_drop_K: float | None = None) -> dict[str, float]:
    """
    Split η·Q_elec into arc surface heat and droplet sensible+latent power.

    ``twin.Q_w`` is the electrical arc power (V·I·PF·duty). Returns dict with
    Q_total, Q_arc, Q_droplet [W] and eta.
    """
    eta = float(twin.eta)
    Q_elec = float(getattr(twin, "Q_w", 0.0) or 0.0)
    Q_total = eta * Q_elec
    if T_drop_K is None:
        from ..solvers.coupled_step import droplet_entry_temperature_K
        T_drop_K = droplet_entry_temperature_K(twin)
    mdot = wire_mass_rate_kg_s(twin)
    q_drop = mdot * (droplet_enthalpy_J_kg(twin, T_drop_K) - ambient_enthalpy_J_kg(twin))
    q_drop = max(0.0, min(q_drop, 0.95 * Q_total))
    q_arc = max(0.0, Q_total - q_drop)
    return {
        "Q_total_W": Q_total,
        "Q_arc_W": q_arc,
        "Q_droplet_W": q_drop,
        "eta": eta,
        "mdot_kg_s": mdot,
        "T_drop_K": float(T_drop_K),
    }


def sigma_arc_m(
    current_A: float,
    voltage_V: float,
    ctwd_m: float,
    *,
    sigma0_m: float = 2.0e-3,
    a_I: float = 5.0e-6,
    a_V: float = 1.0e-5,
    a_ctwd: float = 0.15,
) -> float:
    """
    Fitted arc-radius law σ(I, V, CTWD) — the *only* free process geometry
    knob in the global calibration card (Phase 7).

    σ = σ0 · (1 + a_I·(I−I0) + a_V·(V−V0) + a_ctwd·(CTWD/CTWD0 − 1))
    with I0=180 A, V0=20 V, CTWD0=12 mm as reference.
    """
    I0, V0, ctwd0 = 180.0, 20.0, 0.012
    scale = (
        1.0
        + a_I * (float(current_A) - I0)
        + a_V * (float(voltage_V) - V0)
        + a_ctwd * (float(ctwd_m) / ctwd0 - 1.0)
    )
    return max(0.5e-3, float(sigma0_m) * max(0.3, scale))
