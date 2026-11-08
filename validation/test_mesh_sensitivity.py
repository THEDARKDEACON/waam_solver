"""
test_mesh_sensitivity.py — thermal fusion zone must not depend on dx.

A short Goldak bead (thermal tier, conduction dt) on a 6 mm plate whose top
face is a cell face on both meshes, run at dx = 0.4 and 0.2 mm with the same
physical inputs. Fusion width and depth (liquidus T_max envelope) must
agree within GATE_PCT, for absolute-CF and normalised Goldak.

Pure conduction converges fast, so a large difference here is a mesh-
dependent modelling error (cell-unit parameter, off-by-one-cell placement,
per-cell instead of per-area source), not resolution.

Regression: before the Goldak half-space / arc_k fix, absolute mode gave
W 4.22 → 3.87 mm (9 %) and D 2.19 → 1.97 mm (11 %) from 0.4 to 0.2 mm.
After it: 0.9 % and 2.6 %.

Flow-side mesh convergence is a GPU study: tools/mesh_study.py.
"""

from __future__ import annotations

import copy
import sys

import yaml

from waam_twin.benchmark import measure_fusion_zone_mm
from waam_twin.job import parse_torch_path
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin.torch_path import TorchPathDriver

GATE_PCT = 5.0
MESHES_MM = (0.4, 0.2)
WELD_S = 1.2

_OFF = (
    "enable_vof", "enable_csf_tension", "enable_wetting", "enable_hydrostatic_gravity",
    "enable_bead_freeze", "enable_trailing_solidify", "enable_droplet_impact_pressure",
    "enable_recoil", "use_recoil_clausius_clapeyron", "enable_lorentz", "enable_gas_shear",
    "enable_marangoni", "enable_buoyancy", "enable_arc_pressure", "enable_enthalpy_cap",
    "enable_evaporative_cooling", "arc_surface_weighting",
)

JOB = {
    "simulation": {
        "preset": "minimal", "backend": "cpu", "physics_tier": "thermal",
        "thermal_dt": "auto", "domain_mm": [24, 12, 10], "dx_mm": 0.4,
        "strict_mode": False, **{k: False for k in _OFF},
    },
    "plate": {"thickness_mm": 6.0},
    "material": "materials/validated/ER70S-6.v1.yaml",
    "calibration": None,
    "heat_source": "goldak",
    # bead_calibrate.yaml Goldak lock
    "goldak": {
        "ff": 0.6, "fr": 1.4, "a_front_mm": 2.0, "a_rear_mm": 3.8,
        "b_mm": 2.8, "c_mm": 2.6, "normalize_energy": False,
    },
    "process": {
        "current_A": 91, "voltage_V": 22.0, "arc_efficiency": 0.82,
        "travel_speed_mm_s": 10.0, "wire_feed_m_min": 0.0, "wire_diameter_mm": 1.0,
        "T_ambient_K": 293, "power_factor": 1.0, "duty_cycle": 1.0,
    },
    "torch_path": [
        {"x_mm": 4.0, "y_mm": 6.0, "z_mm": 0.0},
        {"x_mm": 20.0, "y_mm": 6.0, "z_mm": 0.0},
    ],
}


def _fusion(dx_mm: float, normalize: bool) -> tuple[float, float]:
    from waam_twin import WAAMTwin

    reset_taichi()
    init_taichi(backend="cpu")
    job = copy.deepcopy(JOB)
    job["simulation"]["dx_mm"] = float(dx_mm)
    job["goldak"]["normalize_energy"] = bool(normalize)
    tmp = PROJECT_ROOT / ".waam_sweep_tmp" / f"mesh_sens_{dx_mm:g}_{int(normalize)}.yaml"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(yaml.safe_dump(job), encoding="utf-8")
    try:
        twin = WAAMTwin.from_job(tmp)
        twin.reset()
        g = twin.grid
        n = int(round(WELD_S / g.dt))
        driver = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
        for _s, x, y, _z in driver.positions_for_steps(n, g.dt):
            twin.step(x, y, is_welding=True)
        fz = measure_fusion_zone_mm(twin)
    finally:
        tmp.unlink(missing_ok=True)
    return float(fz["fusion_width_mm"]), float(fz["fusion_depth_mm"])


def run() -> float:
    worst = 0.0
    try:
        for normalize in (False, True):
            label = "normalised" if normalize else "absolute"
            res = {dx: _fusion(dx, normalize) for dx in MESHES_MM}
            (w0, d0), (w1, d1) = res[MESHES_MM[0]], res[MESHES_MM[1]]
            dw = abs(w0 - w1) / w1 * 100.0
            dd = abs(d0 - d1) / d1 * 100.0
            worst = max(worst, dw, dd)
            print(
                f"[mesh_sensitivity] Goldak {label:<10} "
                f"dx={MESHES_MM[0]}: W={w0:.3f} D={d0:.3f}  "
                f"dx={MESHES_MM[1]}: W={w1:.3f} D={d1:.3f} mm  "
                f"ΔW={dw:.1f}% ΔD={dd:.1f}%  (gate {GATE_PCT}%)"
            )
            if max(dw, dd) > GATE_PCT:
                raise AssertionError(
                    f"{label} Goldak fusion zone depends on mesh: ΔW={dw:.1f}% "
                    f"ΔD={dd:.1f}% between dx={MESHES_MM[0]} and {MESHES_MM[1]} mm"
                )
    finally:
        # Leave a fresh runtime for whatever runs next in the suite.
        reset_taichi()
        init_taichi(backend="cpu")
    print("[mesh_sensitivity] PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
