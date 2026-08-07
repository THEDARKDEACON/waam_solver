"""
test_cooldown_energy.py — Arc-off cool-down must not invent enthalpy / melt.

Catches two regressions that made liquid lifetime look unrealistically long:
  1. remelt_hot_solid rewriting H from T (creating latent heat)
  2. update_flags_from_phi filling the whole domain base as SOLID metal
"""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi, reset_taichi


def _seed_hot_blob(twin: WAAMTwin) -> tuple[float, float]:
    g = twin.grid
    twin.reset()
    Tl = twin.mat.T_liquidus
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    fl = g.f_l.to_numpy()
    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()

    for i in range(g.nx):
        for j in range(g.ny):
            for k in range(g.nz):
                if flags[i, j, k] == g.FLAG_GAS:
                    continue
                T[i, j, k] = 400.0
                H[i, j, k] = twin.cp_rho * 400.0
                fl[i, j, k] = 0.0
                if k < twin.nz_solid:
                    flags[i, j, k] = g.FLAG_SOLID
                    phi[i, j, k] = 1.0

    ci, cj, ck = g.nx // 2, g.ny // 2, max(1, twin.nz_solid - 1)
    for i in range(ci - 2, ci + 2):
        for j in range(cj - 2, cj + 2):
            for k in range(max(0, ck - 1), ck + 1):
                if flags[i, j, k] == g.FLAG_GAS:
                    continue
                H[i, j, k] = twin.H_liq + twin.cp_rho * 150.0
                T[i, j, k] = Tl + 150.0
                fl[i, j, k] = 1.0
                flags[i, j, k] = g.FLAG_FLUID
                phi[i, j, k] = 1.0

    g.T.from_numpy(T)
    g.H.from_numpy(H)
    g.f_l.from_numpy(fl)
    g.flags.from_numpy(flags.astype(np.int32))
    g.phi.from_numpy(phi.astype(np.float32))
    return ci * g.dx, cj * g.dx


def _metrics(twin: WAAMTwin):
    g = twin.grid
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    fl = g.f_l.to_numpy()
    flags = g.flags.to_numpy()
    metal = flags != g.FLAG_GAS
    return {
        "H_J": float(H[metal].sum() * g.dx**3),
        "n_liq": int((fl > 0.05).sum()),
        "Tpeak_C": float(T[metal].max() - 273.15),
        "n_metal": int(metal.sum()),
    }


def run() -> None:
    reset_taichi()
    init_taichi(backend="cpu")

    root = Path(__file__).resolve().parents[1]
    job = yaml.safe_load((root / "jobs/examples/bead_on_plate.yaml").read_text())
    job["simulation"].update(
        {
            "preset": "minimal",
            "dx_mm": 1.0,
            "domain_mm": [24, 24, 12],
            "enable_lorentz": False,
            "enable_recoil": False,
            "enable_gas_shear": False,
            "enable_droplet_impact_pressure": False,
            "enable_csf_tension": False,
            "enable_wetting": False,
            "enable_hydrostatic_gravity": False,
            "enable_evaporative_cooling": False,
            "enable_vof": True,
            "enable_bead_freeze": True,
        }
    )
    job["plate"] = {"size_mm": [12, 12], "thickness_mm": 4.0}
    job["material"] = str(root / "materials/validated/ER70S-6.v1.yaml")
    job["torch_path_csv"] = str(root / "jobs/paths/bead_line.csv")
    tmp = Path("/tmp/test_cooldown_energy.yaml")
    tmp.write_text(yaml.dump(job))

    twin = WAAMTwin.from_job(str(tmp))
    for attr in (
        "enable_lorentz",
        "enable_recoil",
        "enable_gas_shear",
        "enable_droplet_impact_pressure",
        "enable_csf_tension",
        "enable_wetting",
        "enable_hydrostatic_gravity",
        "enable_evaporative_cooling",
    ):
        setattr(twin, attr, False)

    g = twin.grid
    cx, cy = _seed_hot_blob(twin)
    m0 = _metrics(twin)

    for _ in range(int(0.5 / g.dt)):
        twin.step(cx, cy, is_welding=False)
    m1 = _metrics(twin)

    # Must not invent a full-domain plate (12×12×4 → 24×24×4).
    assert m1["n_metal"] <= m0["n_metal"] + 64, (
        f"VOF/flags invented metal: {m0['n_metal']} → {m1['n_metal']}"
    )
    # Arc-off: total metal enthalpy must not grow (allow tiny float noise).
    assert m1["H_J"] <= m0["H_J"] * 1.02 + 5.0, (
        f"enthalpy grew during cool-down: {m0['H_J']:.1f} → {m1['H_J']:.1f} J"
    )
    # Small blob should solidify well before 0.5 s (pure conduction ~0.2 s).
    assert m1["n_liq"] == 0, f"liquid still present after 0.5 s: n_liq={m1['n_liq']}"
    assert m1["Tpeak_C"] < twin.mat.T_solidus - 273.15, (
        f"T_peak still at/above solidus: {m1['Tpeak_C']:.1f} C"
    )

    print(
        f"[cooldown_energy] H {m0['H_J']:.1f}→{m1['H_J']:.1f} J  "
        f"nliq {m0['n_liq']}→{m1['n_liq']}  "
        f"Tpeak {m0['Tpeak_C']:.0f}→{m1['Tpeak_C']:.0f} C  "
        f"nmetal {m0['n_metal']}→{m1['n_metal']} OK"
    )


if __name__ == "__main__":
    run()
    print("PASS")
