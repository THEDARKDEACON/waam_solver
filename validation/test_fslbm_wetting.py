"""
test_fslbm_wetting.py — static contact angle on the free-surface LBM path.

A hemispherical liquid drop (R = 6 cells, the 90° cap) sits on a plate held
just below solidus, with Marangoni and gravity off, so only capillarity acts.
Under the free-surface LBM the contact angle enters through the interface
curvature (θ-corrected wall normals in κ = −∇·n̂). Checks after 600 steps:

  1. θ = 45° spreads to the spherical-cap contact radius (±10 %).
  2. θ = 135° stays clearly narrower and taller than θ = 45°.
  3. With surface tension off the drop does not move and keeps its mass.

Regression: fs_stream used the mirrored-φ (90°) wall curvature only, so
``contact_angle_deg`` had no effect on production (FSLBM) jobs.

Known limitation (not asserted): with surface tension on, every θ drifts
slowly toward spreading over thousands of steps (θ = 90° goes from the
hemisphere's r = 6 to r ≈ 7 cells by step 3000) and the drop loses ~2 % of
its mass at the contact line. With surface tension off neither happens, so
the source is the staircase contact-line curvature, not the mass exchange.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.physics.heat_source import wire_volumetric_enthalpy_J_m3
from waam_twin.runtime import init_taichi

R0 = 6.0
STEPS = 600


def _cap_radius(volume: float, theta_deg: float) -> float:
    """Contact radius of a spherical cap of the given volume and angle."""
    th = math.radians(theta_deg)
    c = math.cos(th)
    R = (3.0 * volume / (math.pi * (2.0 + c) * (1.0 - c) ** 2)) ** (1.0 / 3.0)
    return R * math.sin(th)


def _drop(theta_deg: float, csf: bool = True, steps: int = STEPS):
    twin = WAAMTwin(
        nx=40, ny=40, nz=24, dx=2.5e-4,
        enable_vof=True, enable_csf_tension=csf, enable_wetting=True,
        contact_angle_deg=theta_deg, enable_hydrostatic_gravity=False,
        enable_bead_freeze=False, max_tracers=10,
    )
    twin.use_free_surface_lbm = True
    twin.enable_marangoni = False
    twin.reset()
    g = twin.grid
    g.ensure_fslbm_fields()
    k0 = twin.nz_solid

    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()
    H = g.H.to_numpy()
    T = g.T.to_numpy()
    f_l = g.f_l.to_numpy()
    T_drop = float(twin.mat.T_liquidus) + 200.0
    # Plate just below solidus so the drop stays liquid for the whole run.
    H[flags == g.FLAG_SOLID] = wire_volumetric_enthalpy_J_m3(twin, float(twin.mat.T_solidus) - 20.0)

    ci, cj = g.nx / 2.0, g.ny / 2.0
    I, J, K = np.meshgrid(
        np.arange(g.nx) + 0.5, np.arange(g.ny) + 0.5, np.arange(g.nz) + 0.5, indexing="ij",
    )
    d = np.sqrt((I - ci) ** 2 + (J - cj) ** 2 + (K - k0) ** 2)
    fill = np.clip(R0 + 0.5 - d, 0.0, 1.0) * (K > k0)
    solid = flags == g.FLAG_SOLID
    fluid = (fill >= 0.999) & ~solid
    iface = (fill > 0.0) & ~fluid & ~solid
    gas = ~solid & ~fluid & ~iface
    flags[gas] = g.FLAG_GAS
    phi[gas] = 0.0
    flags[fluid] = g.FLAG_FLUID
    phi[fluid] = 1.0
    flags[iface] = g.FLAG_IFACE
    phi[iface] = fill[iface]
    liq = fluid | iface
    H[liq] = wire_volumetric_enthalpy_J_m3(twin, T_drop)
    T[liq] = T_drop
    f_l[liq] = 1.0
    g.flags.from_numpy(flags.astype(np.int32))
    g.phi.from_numpy(phi.astype(np.float32))
    g.H.from_numpy(H.astype(np.float32))
    g.T.from_numpy(T.astype(np.float32))
    g.f_l.from_numpy(f_l.astype(np.float32))

    def metrics() -> tuple[float, float, float]:
        p = g.phi.to_numpy()
        fl = g.flags.to_numpy()
        m = np.where(fl != g.FLAG_SOLID, p, 0.0)
        r_base = float(np.sqrt((m[:, :, k0] > 0.5).sum() / np.pi))
        h = float(m[int(ci), int(cj), k0:].sum())
        return r_base, h, float(m[:, :, k0:].sum())

    r_start, h_start, v0 = metrics()
    for _ in range(steps):
        twin.step(ci * g.dx, cj * g.dx, is_welding=False)
    r, h, v = metrics()
    return r_start, r, h, v0, v


def run() -> None:
    init_taichi(backend="cpu")

    _rs, r45, h45, v0, _v = _drop(45.0)
    r_cap = _cap_radius(v0, 45.0)
    print(f"[fslbm_wetting] θ=45°  r={r45:.2f} h={h45:.2f} cells  cap r={r_cap:.2f}")
    if abs(r45 - r_cap) > 0.10 * r_cap:
        raise AssertionError(
            f"θ=45° contact radius {r45:.2f} not within 10% of the cap {r_cap:.2f}"
        )

    _rs, r135, h135, _v0, _v = _drop(135.0)
    print(f"[fslbm_wetting] θ=135° r={r135:.2f} h={h135:.2f} cells  (cap r={_cap_radius(v0, 135.0):.2f})")
    if not (r135 < 0.75 * r45 and h135 > 1.5 * h45):
        raise AssertionError(
            f"contact angle inert on FSLBM: θ=135° r={r135:.2f} h={h135:.2f} "
            f"vs θ=45° r={r45:.2f} h={h45:.2f}"
        )

    r_start, r0, h0, m0, m1 = _drop(90.0, csf=False)
    drift = abs(m1 - m0) / m0
    print(f"[fslbm_wetting] γ=0   r {r_start:.2f}→{r0:.2f}  mass drift {drift:.1e}")
    if abs(r0 - r_start) > 1.0 or drift > 1e-3:
        raise AssertionError(
            f"force-free drop moved or lost mass: r {r_start:.2f}→{r0:.2f}, drift {drift:.1e}"
        )
    print("[fslbm_wetting] PASS")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
