"""
test_force_direction_gate.py — Directional physics gate (Lorentz / S-ppm).

PHYSICS_FORCE_CORRECTNESS_SPEC §11: seeded-pool smoke so absolute melt-in
time is not required. Asserts:
  - Lorentz ON → nonzero J×B and ≥ OFF downward pumping
  - Sahoo high-S dγ/dT > 0, low-S < 0; high-S not both shallower and wider
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.job import apply_physics_tier
from waam_twin.physics.surfactant import apply_sahoo_to_material_props, dgamma_dT_sahoo
from waam_twin.platform import init_taichi
from waam_twin.tools.force_ablation import _seed_pool


def _short_weld(
    *,
    enable_lorentz: bool,
    sulphur_ppm: float | None = None,
    n_steps: int = 50,
) -> dict:
    twin = WAAMTwin(
        nx=36, ny=24, nz=22, dx=3.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        heat_source="gaussian2d",
        arc_power_W=3500.0,
        arc_efficiency=0.8,
        welding_current_A=200.0,
        travel_speed_m_s=0.004,
        droplet_freq_hz=40.0,
        max_tracers=20,
        lorentz_jacobi_iters=80,
    )
    twin.wire_feed_m_s = 0.5 / 60.0
    apply_physics_tier(twin, "full")
    twin.enable_lorentz = enable_lorentz
    if enable_lorentz:
        twin.grid.ensure_lorentz_fields()
    twin.enable_force_diagnostics = True
    twin.reset()
    _seed_pool(twin, radius=5)

    if sulphur_ppm is not None:
        twin.mat.sulphur_ppm = float(sulphur_ppm)
        apply_sahoo_to_material_props(twin.mat)
        twin.use_material_tables = False
        twin.dgamma_dT_lu = twin.mat.dgamma_dT * twin.force_scale / twin.grid.dx
        twin.gamma_lu = twin.mat.gamma_0 * twin.force_scale / twin.grid.dx

    g = twin.grid
    x0 = (g.nx // 3) * g.dx
    y0 = (g.ny // 2) * g.dx
    for _ in range(n_steps):
        twin.step(x0, y0, is_welding=True)
        x0 += twin.travel_speed_m_s * g.dt

    telem = twin.get_telemetry()
    uz = g.uz.to_numpy()
    fl = g.f_l.to_numpy()
    mask = fl > 0.5
    n_liq = int(mask.sum())
    uz_down = float((-uz[mask]).max()) * g.dx / g.dt if n_liq else 0.0
    # Penetration proxy: lowest liquid k below substrate top
    if n_liq:
        ks = np.where(mask)[2]
        depth_cells = max(0, twin.nz_solid - int(ks.min()))
        depth_mm = depth_cells * g.dx * 1000.0
        js = np.where(mask)[1]
        width_mm = (int(js.max()) - int(js.min()) + 1) * g.dx * 1000.0
    else:
        depth_mm = width_mm = 0.0
    return {
        "depth_mm": depth_mm,
        "width_mm": width_mm,
        "n_liquid": n_liq,
        "uz_down_ms": uz_down,
        "dgamma_dT": twin.mat.dgamma_dT,
        "force_diagnostics": telem.get("force_diagnostics", {}),
    }


def run() -> None:
    init_taichi(backend="cpu")

    off = _short_weld(enable_lorentz=False, n_steps=45)
    on = _short_weld(enable_lorentz=True, n_steps=45)
    print(
        f"[direction] Lorentz OFF D={off['depth_mm']:.3f} uz↓={off['uz_down_ms']:.4f}  "
        f"ON D={on['depth_mm']:.3f} uz↓={on['uz_down_ms']:.4f}  "
        f"Lz={on['force_diagnostics'].get('f_lorentz_max', 0):.3e}"
    )
    if on["n_liquid"] < 5 or off["n_liquid"] < 5:
        raise AssertionError("Lorentz gate: insufficient liquid for comparison")
    if on["force_diagnostics"].get("f_lorentz_max", 0) <= 0:
        raise AssertionError("Lorentz ON but f_lorentz_max is zero")
    deeper = on["depth_mm"] + 1e-6 >= off["depth_mm"]
    stronger_down = on["uz_down_ms"] >= off["uz_down_ms"] * 0.8
    if not (deeper or stronger_down):
        raise AssertionError(
            "Lorentz ON should deepen pool or strengthen downward pumping vs OFF"
        )

    assert dgamma_dT_sahoo(1850.0, 20.0) < 0
    assert dgamma_dT_sahoo(1850.0, 150.0) > 0

    low_s = _short_weld(enable_lorentz=False, sulphur_ppm=20.0, n_steps=45)
    high_s = _short_weld(enable_lorentz=False, sulphur_ppm=150.0, n_steps=45)
    print(
        f"[direction] low-S dγ/dT={low_s['dgamma_dT']:.2e} D={low_s['depth_mm']:.3f} W={low_s['width_mm']:.3f}  "
        f"high-S dγ/dT={high_s['dgamma_dT']:.2e} D={high_s['depth_mm']:.3f} W={high_s['width_mm']:.3f}"
    )
    if low_s["dgamma_dT"] >= 0:
        raise AssertionError("low-S should have dγ/dT < 0 (outward)")
    if high_s["dgamma_dT"] <= 0:
        raise AssertionError("high-S should have dγ/dT > 0 (inward)")
    if (
        high_s["depth_mm"] + 0.3 < low_s["depth_mm"]
        and high_s["width_mm"] > low_s["width_mm"] + 0.5
    ):
        raise AssertionError(
            "high-S (inward) should not be both shallower and wider than low-S"
        )
    print("[direction] Lorentz + S-ppm gates OK")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
