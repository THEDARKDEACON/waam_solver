"""
test_fslbm_fill_conservation.py — freeze/remelt keep the free-surface fill,
and droplets enter at their nominal temperature.

1. Under the free-surface LBM, phi is an interface cell's fill (its mass).
   solidify_cooled_metal and remelt_hot_solid used to force phi = 1, which
   created up to (1 − ε) of a cell of metal (and (1 − ε)·H of energy) per
   freeze or remelt of a surface cell. With fs_iface ≥ 0 the fill must be
   kept and a partly filled cell must reopen as IFACE. The legacy VOF path
   (fs_iface = −1) still snaps phi to 1.
2. Droplet enthalpy (wire_volumetric_enthalpy_J_m3) must invert to T_drop on
   both the variable-cp and the scalar phase paths. The old
   cp·T_sol + L + cp·(T − T_liq) form arrived ~60 K cold for ER70S-6.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.paths import PROJECT_ROOT
from waam_twin.physics import free_surface
from waam_twin.physics.heat_source import wire_volumetric_enthalpy_J_m3
from waam_twin.runtime import init_taichi
from waam_twin.solvers.coupled_step import _update_phase

MATERIAL = str(PROJECT_ROOT / "materials" / "validated" / "ER70S-6.v1.yaml")
FILL = 0.9


def _set_cell(field, idx, value) -> None:
    a = field.to_numpy()
    a[idx] = value
    field.from_numpy(a)


def _freeze_then_remelt(twin: WAAMTwin, fs_iface: int) -> tuple[int, float, int, float]:
    """Freeze a cold partly filled metal cell, then remelt it. Return flags/phi after each."""
    g = twin.grid
    twin.reset(test_fluid_domain=True)
    c = (g.nx // 2, g.ny // 2, g.nz // 2)
    start_flag = g.FLAG_IFACE if fs_iface >= 0 else g.FLAG_FLUID
    _set_cell(g.flags, c, start_flag)
    _set_cell(g.phi, c, FILL)
    _set_cell(g.f_l, c, 0.0)
    _set_cell(g.T, c, float(twin.mat.T_solidus) - 100.0)
    _set_cell(g.H, c, float(twin.H_sol) * 0.9)

    free_surface.solidify_cooled_metal(
        g.T, g.H, g.f_l, g.phi, g.flags, g.ux, g.uy, g.uz,
        twin.H_sol, twin.mat.T_solidus, True,
        g.FLAG_SOLID, g.FLAG_FLUID, g.FLAG_GAS,
        fs_iface=fs_iface,
    )
    flag_frozen = int(g.flags.to_numpy()[c])
    phi_frozen = float(g.phi.to_numpy()[c])

    _set_cell(g.H, c, float(twin.H_liq + 0.1 * twin.L_rho))
    free_surface.remelt_hot_solid(
        g.T, g.H, g.f_l, g.phi, g.flags,
        g.f_src, g.rho, g.ux, g.uy, g.uz, g.w,
        twin.L_rho, twin.H_sol, twin.H_liq,
        g.FLAG_SOLID, g.FLAG_FLUID,
        fs_iface=fs_iface,
    )
    flag_remelt = int(g.flags.to_numpy()[c])
    phi_remelt = float(g.phi.to_numpy()[c])
    return flag_frozen, phi_frozen, flag_remelt, phi_remelt


def _check_fill(twin: WAAMTwin) -> None:
    g = twin.grid
    fz, pz, fr, pr = _freeze_then_remelt(twin, int(g.FLAG_IFACE))
    print(f"[fslbm_fill] FSLBM : frozen flag={fz} phi={pz:.3f} → remelt flag={fr} phi={pr:.3f}")
    if fz != g.FLAG_SOLID:
        raise AssertionError(f"cold interface cell did not freeze (flag={fz})")
    if abs(pz - FILL) > 1e-6:
        raise AssertionError(f"freeze changed FSLBM fill {FILL} → {pz} (metal created)")
    if fr != g.FLAG_IFACE:
        raise AssertionError(f"partly filled cell should remelt as IFACE, got flag={fr}")
    if abs(pr - FILL) > 1e-6:
        raise AssertionError(f"remelt changed FSLBM fill {FILL} → {pr} (metal created)")

    fz, pz, fr, pr = _freeze_then_remelt(twin, -1)
    print(f"[fslbm_fill] legacy: frozen flag={fz} phi={pz:.3f} → remelt flag={fr} phi={pr:.3f}")
    if fz != g.FLAG_SOLID or abs(pz - 1.0) > 1e-6:
        raise AssertionError("legacy VOF freeze must still snap phi to 1")
    if fr != g.FLAG_FLUID or abs(pr - 1.0) > 1e-6:
        raise AssertionError("legacy VOF remelt must still open FLUID with phi = 1")


def _recovered_T(twin: WAAMTwin, H_vol: float) -> float:
    g = twin.grid
    g.H.fill(float(H_vol))
    _update_phase(twin, g)
    return float(np.mean(g.T.to_numpy()))


def _check_droplet_enthalpy(twin: WAAMTwin) -> float:
    mat = twin.mat
    T_drop = float(mat.T_liquidus) + 400.0
    worst = 0.0
    for label, use_tables in (("variable-cp", True), ("scalar", False)):
        saved = twin.use_material_tables
        twin.use_material_tables = use_tables
        try:
            H_new = wire_volumetric_enthalpy_J_m3(twin, T_drop)
            H_old = (
                twin.cp_rho * mat.T_solidus + twin.L_rho
                + twin.cp_rho * (T_drop - mat.T_liquidus)
            )
            T_new = _recovered_T(twin, H_new)
            T_old = _recovered_T(twin, H_old)
        finally:
            twin.use_material_tables = saved
        err = abs(T_new - T_drop)
        worst = max(worst, err)
        print(
            f"[fslbm_fill] droplet {label:<11} T_drop={T_drop:.1f} K  "
            f"recovered={T_new:.2f} K (old formula {T_old:.2f} K)"
        )
        if err > 1.0:
            raise AssertionError(
                f"{label}: droplet enthalpy recovers {T_new:.2f} K, expected {T_drop:.1f} K"
            )
    return worst


def run() -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        material=MATERIAL,
        nx=8, ny=8, nz=8, dx=5e-4,
        enable_vof=True, enable_bead_freeze=True,
        C_darcy=0.0, max_tracers=10,
    )
    if not twin.use_material_tables:
        raise AssertionError(f"{MATERIAL} should load cp tables")
    _check_fill(twin)
    err = _check_droplet_enthalpy(twin)
    print("[fslbm_fill] PASS")
    return err


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    raise SystemExit(0)
