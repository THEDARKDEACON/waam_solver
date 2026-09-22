"""
test_dual_alloy.py — Plate YAML vs wire YAML on birth-tagged cells.

Same-alloy jobs (no plate.material) must stay on the single-alloy path so
the ER70S-6 calibrate lock is unchanged.
"""

from __future__ import annotations

import sys

from waam_twin import WAAMTwin
from waam_twin.job import load_job_config
from waam_twin.kernels import ALLOY_PLATE, ALLOY_WIRE
from waam_twin.materials import load_material
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.solvers.coupled_step import _refresh_thermal, _update_phase


def _mushy_mid(H_sol: float, H_liq: float) -> float:
    return 0.5 * (H_sol + H_liq)


def run() -> None:
    reset_taichi()
    init_taichi(backend="cpu")
    wire = load_material("materials/placeholders/ER70S-6.yaml")
    plate = load_material("materials/placeholders/SS316L.yaml")
    if abs(wire.T_solidus - plate.T_solidus) < 20.0:
        raise AssertionError("test alloys must have distinct melting ranges")

    twin = WAAMTwin(
        material=wire,
        plate_material=plate,
        nx=16,
        ny=12,
        nz=14,
        dx=8e-4,
        max_tracers=8,
    )
    if not twin.use_dual_alloy:
        raise AssertionError("expected dual-alloy path for ER70S-6 wire / SS316L plate")
    twin.reset()
    g = twin.grid
    aid = g.alloy_id.to_numpy()
    flags = g.flags.to_numpy()
    i0, i1, j0, j1 = twin.resolve_plate_ij()
    ip = (i0 + i1) // 2
    jp = (j0 + j1) // 2
    kp = max(0, twin.nz_solid - 2)
    if int(aid[ip, jp, kp]) != ALLOY_PLATE:
        raise AssertionError(f"plate cell alloy_id={aid[ip, jp, kp]} want {ALLOY_PLATE}")
    if int(flags[ip, jp, kp]) != g.FLAG_SOLID:
        raise AssertionError("plate cell should be SOLID after reset")
    kg = min(g.nz - 1, twin.nz_solid + 2)
    if int(aid[ip, jp, kg]) != ALLOY_WIRE:
        raise AssertionError("gas/deposit cells should be wire alloy 0")

    H = g.H.to_numpy()
    H[ip, jp, kp] = _mushy_mid(twin.plate_H_sol, twin.plate_H_liq)
    iw, jw, kw = ip, jp, min(g.nz - 2, twin.nz_solid + 1)
    flags[iw, jw, kw] = g.FLAG_FLUID
    aid[iw, jw, kw] = ALLOY_WIRE
    H[iw, jw, kw] = _mushy_mid(twin.H_sol, twin.H_liq)
    g.H.from_numpy(H)
    g.flags.from_numpy(flags)
    g.alloy_id.from_numpy(aid)
    g.phi[iw, jw, kw] = 1.0
    g.f_l[iw, jw, kw] = 1.0

    _refresh_thermal(twin, g)
    _update_phase(twin, g)

    T = g.T.to_numpy()
    fl = g.f_l.to_numpy()
    Tmid_p = 0.5 * (plate.T_solidus + plate.T_liquidus)
    Tmid_w = 0.5 * (wire.T_solidus + wire.T_liquidus)
    if abs(float(T[ip, jp, kp]) - Tmid_p) > 8.0:
        raise AssertionError(
            f"plate mushy T={T[ip, jp, kp]:.1f}K not near plate mid {Tmid_p:.1f}K"
        )
    if abs(float(fl[ip, jp, kp]) - 0.5) > 0.08:
        raise AssertionError(f"plate mushy f_l={fl[ip, jp, kp]:.3f} want ~0.5")
    if abs(float(T[iw, jw, kw]) - Tmid_w) > 8.0:
        raise AssertionError(
            f"wire mushy T={T[iw, jw, kw]:.1f}K not near wire mid {Tmid_w:.1f}K"
        )
    if int(g.alloy_id.to_numpy()[ip, jp, kp]) != ALLOY_PLATE:
        raise AssertionError("melted plate cell must keep birth alloy")
    if abs(float(g.alloy_frac.to_numpy()[ip, jp, kp]) - 1.0) > 1e-6:
        raise AssertionError("mixing off: melted plate must keep plate composition")

    alpha = g.alpha_lu_field.to_numpy()
    if abs(float(alpha[ip, jp, kp]) - float(twin.plate_alpha_lu)) > 1e-9:
        raise AssertionError(
            f"plate α_lu={alpha[ip, jp, kp]:.6e} want {twin.plate_alpha_lu:.6e}"
        )
    if abs(twin.plate_alpha_lu - twin.alpha_lu) < 1e-12:
        raise AssertionError("plate and wire α_lu should differ for this couple")

    same = WAAMTwin(material=wire, nx=8, ny=8, nz=8, dx=8e-4, max_tracers=4)
    if same.use_dual_alloy:
        raise AssertionError("omitting plate_material must keep single-alloy path")

    job = load_job_config(PROJECT_ROOT / "jobs/examples/bead_dissimilar_ss316l_plate.yaml")
    plate_spec = (job.get("plate") or {}).get("material")
    if not plate_spec or "SS316L" not in str(plate_spec):
        raise AssertionError(f"example job missing plate.material, got {plate_spec!r}")
    if "ER70S-6" not in str(job.get("material", "")):
        raise AssertionError("example job wire material should be ER70S-6")

    print(
        f"[dual_alloy] plate T_sol={plate.T_solidus:.0f}K  wire T_sol={wire.T_solidus:.0f}K  "
        f"plate mushy T={T[ip, jp, kp]:.1f}K  wire mushy T={T[iw, jw, kw]:.1f}K"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
