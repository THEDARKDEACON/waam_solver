"""
Surfactant (S/O) effect on surface tension and dγ/dT.

Two models:

1. **heiple** — static S-ppm scale of a base dγ/dT (Mills & Keene 1998 /
   Heiple–Roper thresholds). Fast; no T-local chemistry.

2. **sahoo** — Sahoo, DebRoy & McNallan (1988) Fe–S isotherm:

       γ(T,a_S) = γ_m − A(T − T_m) − R T Γ_s ln(1 + K a_S)

       K = k_1 exp(−ΔH⁰ / R T)

       dγ/dT from analytic differentiation (ΔH_mix ≈ 0).

   Activity a_S ≈ wt% S = sulphur_ppm / 1e4 (dilute welding approximation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Heiple / Mills static scale ──────────────────────────────────────────────

def surfactant_dgamma_dT_scale(sulphur_ppm: float) -> float:
    """
    Return multiplier for base dγ/dT preserving sign convention.

    <30 ppm: outward (negative dγ/dT) — scale magnitude ~1.0
    >60 ppm: inward tendency — flip to positive fraction of |base|
    30–60 ppm: linear blend
    """
    s = max(0.0, sulphur_ppm)
    if s <= 30.0:
        return 1.0
    if s >= 60.0:
        return -0.35
    t = (s - 30.0) / 30.0
    return 1.0 - 1.35 * t


def effective_dgamma_dT(base_dgamma_dT: float, sulphur_ppm: float) -> float:
    """Apply Heiple surfactant scaling; base is negative for low-S carbon steel."""
    scale = surfactant_dgamma_dT_scale(sulphur_ppm)
    if scale < 0:
        return -abs(base_dgamma_dT) * scale
    return base_dgamma_dT * scale


def scale_dgamma_table(
    table: list[tuple[float, float]],
    sulphur_ppm: float,
) -> list[tuple[float, float]]:
    """Scale T-dependent dγ/dT table entries (Heiple model)."""
    scale = surfactant_dgamma_dT_scale(sulphur_ppm)
    out: list[tuple[float, float]] = []
    for t_k, dg in table:
        if scale < 0:
            out.append((t_k, -abs(dg) * scale))
        else:
            out.append((t_k, dg * scale))
    return out


# ── Sahoo / DebRoy / McNallan (1988) Fe–S ────────────────────────────────────

@dataclass(frozen=True)
class SahooDebRoyParams:
    """
    Fe–S parameters from Sahoo et al., Metall. Trans. B 19 (1988) 483–491.

    Units: mol basis (R = 8.314 J/(mol·K), Γ_s in mol/m², ΔH⁰ in J/mol).
    """

    gamma_m: float = 1.943          # N/m at T_m (pure Fe)
    A: float = 4.3e-4               # N/(m·K); −dγ/dT for pure metal
    T_m: float = 1809.0             # K
    Gamma_s: float = 1.3e-5         # mol/m²  (= 1.3e-8 kmol/m²)
    k1: float = 0.00318             # —
    delta_H0: float = -1.66e5       # J/mol   (= −1.66e8 J/kmol)
    R: float = 8.314462618          # J/(mol·K)


FE_S_SAHOO = SahooDebRoyParams()

# Fe–O: Sahoo, DebRoy & McNallan (1988), Table II — Γ_s 2.03e-8 kmol/m²,
# k1 1.38e-2, ΔH⁰ −1.463e8 J/kmol. Oxygen comes from the shielding gas
# (Ar–CO₂ / Ar–O₂), not the wire certificate, so it is a process input.
FE_O_SAHOO = SahooDebRoyParams(Gamma_s=2.03e-5, k1=1.38e-2, delta_H0=-1.463e5)


def _gamma_s_o(T: float, sulphur_ppm: float, oxygen_ppm: float, T_m: float) -> float:
    """γ(T) with S and O competing for the same surface sites.

    γ = γ_m − A(T − T_m) − R T Γ̄ ln(1 + K_S a_S + K_O a_O),
    Γ̄ the coverage-weighted saturation excess (McNallan & DebRoy 1991 form;
    reduces exactly to the single-solute Sahoo isotherm when either is 0).
    """
    pS, pO = FE_S_SAHOO, FE_O_SAHOO
    T = max(float(T), 300.0)
    kS = _adsorption_K(T, pS) * sulphur_ppm_to_activity(sulphur_ppm)
    kO = _adsorption_K(T, pO) * sulphur_ppm_to_activity(oxygen_ppm)
    tot = kS + kO
    G = (pS.Gamma_s * kS + pO.Gamma_s * kO) / tot if tot > 0.0 else 0.0
    return pS.gamma_m - pS.A * (T - T_m) - pS.R * T * G * math.log(1.0 + tot)


def sulphur_ppm_to_activity(sulphur_ppm: float) -> float:
    """Dilute Fe–S: a_S ≈ wt% S = ppm / 10000."""
    return max(0.0, float(sulphur_ppm)) / 10000.0


def _adsorption_K(T: float, p: SahooDebRoyParams) -> float:
    T = max(float(T), 300.0)
    return p.k1 * math.exp(-p.delta_H0 / (p.R * T))


def gamma_sahoo(
    T: float,
    sulphur_ppm: float,
    params: SahooDebRoyParams | None = None,
    T_m: float | None = None,
) -> float:
    """
    Surface tension γ(T, a_S) [N/m] — Sahoo/DebRoy eq. (6)/(7).
    """
    p = params or FE_S_SAHOO
    Tm = float(T_m) if T_m is not None else p.T_m
    T = max(float(T), 300.0)
    a_s = sulphur_ppm_to_activity(sulphur_ppm)
    K = _adsorption_K(T, p)
    Ka = K * a_s
    return (
        p.gamma_m
        - p.A * (T - Tm)
        - p.R * T * p.Gamma_s * math.log(1.0 + Ka)
    )


def dgamma_dT_sahoo(
    T: float,
    sulphur_ppm: float,
    params: SahooDebRoyParams | None = None,
    oxygen_ppm: float = 0.0,
) -> float:
    """
    Analytic dγ/dT [N/(m·K)] for Sahoo/DebRoy (ΔH_mix ≈ 0).

        dγ/dT = −A − R Γ_s ln(1+K a_S) − [K a_S/(1+K a_S)] Γ_s ΔH⁰ / T

    With ``oxygen_ppm`` > 0 the S+O competitive isotherm is differentiated
    numerically (central difference, 0.5 K).
    """
    if oxygen_ppm and oxygen_ppm > 0.0:
        h = 0.5
        Tm = FE_S_SAHOO.T_m
        return (
            _gamma_s_o(T + h, sulphur_ppm, oxygen_ppm, Tm)
            - _gamma_s_o(T - h, sulphur_ppm, oxygen_ppm, Tm)
        ) / (2.0 * h)
    p = params or FE_S_SAHOO
    T = max(float(T), 300.0)
    a_s = sulphur_ppm_to_activity(sulphur_ppm)
    K = _adsorption_K(T, p)
    Ka = K * a_s
    term_ln = p.R * p.Gamma_s * math.log(1.0 + Ka)
    frac = Ka / (1.0 + Ka) if Ka > 0.0 else 0.0
    term_ads = frac * p.Gamma_s * p.delta_H0 / T
    return -p.A - term_ln - term_ads


def build_sahoo_dgamma_table(
    sulphur_ppm: float,
    T_min: float,
    T_max: float,
    n: int = 25,
    params: SahooDebRoyParams | None = None,
    oxygen_ppm: float = 0.0,
) -> list[tuple[float, float]]:
    """Build (T, dγ/dT) knots for GPU material tables."""
    n = max(2, int(n))
    T_lo = float(T_min)
    T_hi = max(T_lo + 1.0, float(T_max))
    out: list[tuple[float, float]] = []
    for i in range(n):
        t = T_lo + (T_hi - T_lo) * i / (n - 1)
        out.append((t, dgamma_dT_sahoo(t, sulphur_ppm, params, oxygen_ppm)))
    return out


def apply_sahoo_to_material_props(props) -> None:
    """
    Replace props.dgamma_dT / tables.dgamma_dT with Sahoo/DebRoy Fe–S values
    (Fe–S–O when ``props.oxygen_ppm`` > 0).

    Mutates ``props`` in place. Uses alloy T_liquidus as melt reference for γ.
    """
    T_liq = float(props.T_liquidus)
    T_hi = max(T_liq + 800.0, 2600.0)
    o_ppm = float(getattr(props, "oxygen_ppm", 0.0) or 0.0)
    table = build_sahoo_dgamma_table(props.sulphur_ppm, T_liq, T_hi, n=8, oxygen_ppm=o_ppm)
    props.tables.dgamma_dT = table
    # Representative scalar used when tables are disabled
    props.dgamma_dT = dgamma_dT_sahoo(T_liq + 50.0, props.sulphur_ppm, oxygen_ppm=o_ppm)
    if o_ppm > 0.0:
        props.gamma_0 = _gamma_s_o(T_liq, props.sulphur_ppm, o_ppm, T_liq)
    else:
        props.gamma_0 = gamma_sahoo(T_liq, props.sulphur_ppm, T_m=T_liq)
    props.T_ref_gamma = T_liq


def apply_surfactant_model(props, model: str = "heiple") -> None:
    """
    Apply surfactant chemistry to loaded material props.

    model:
      - ``heiple`` — static ppm scale of YAML dγ/dT (default, backward compatible)
      - ``sahoo`` / ``sahoo_debroy`` / ``debroy`` — local T-table from Sahoo/DebRoy
    """
    key = (model or "heiple").lower().replace("-", "_")
    if key in ("sahoo", "sahoo_debroy", "debroy"):
        apply_sahoo_to_material_props(props)
        return
    # Heiple path
    props.dgamma_dT = effective_dgamma_dT(props.dgamma_dT, props.sulphur_ppm)
    if props.tables.dgamma_dT:
        props.tables.dgamma_dT = scale_dgamma_table(
            props.tables.dgamma_dT, props.sulphur_ppm,
        )


def lorentz_reference_accel_m_s2(current_A: float, pool_radius_m: float, rho_kg_m3: float) -> float:
    """
    Order-of-magnitude Lorentz acceleration |J×B|/ρ for a GTA pool.

    J ~ I/(πr²), B ~ μ₀I/(2πr) → |J×B| ~ μ₀I²/(2π²r³) [N/m³] (Kou/Szekely).
    (A previous version used r² in the denominator — that is the magnetic
    PRESSURE μ₀I²/4π²r² [N/m²], off by 1/r ≈ 500× at pool scale.)
    """
    mu0 = 4.0e-7 * math.pi
    r = max(pool_radius_m, 1.0e-4)
    f_vol = mu0 * current_A * current_A / (2.0 * math.pi * math.pi * r ** 3)
    return f_vol / max(rho_kg_m3, 1.0)
