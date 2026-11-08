"""
test_fslbm_wetting.py — static contact angle on the free-surface LBM.

Isothermal sessile drop on an inert wall (no thermal step, so the wall
cannot melt), driven by collide + _free_surface_stream as in test_fslbm.
The contact angle enters through the interface curvature (θ-corrected wall
normals); contact-line slip (fs_contact_slip) lets the line move.

Each target θ is approached from a flatter and a steeper spherical cap.
Checks:
  1. θ = 90° (the starting hemisphere) is held within 4°.
  2. For θ = 45° and 135° the two end states bracket the target
     (apparent angle from the cap h/r), with a band no wider than 20°.
  3. Metal mass Σφρ drifts < 1e-3.

Regressions caught:
  - fs_stream used the mirrored-φ (90°) wall: every θ ended near 90°.
  - no-slip contact line: 45° approached from 90° pinned at ~61°
    (band 36–61°), 135° from 90° at ~112°.

Known limitation: a ±8–9° hysteresis band remains (contact-line cell
conversion thresholds).
"""

from __future__ import annotations

import math
import sys

import numpy as np

from waam_twin import WAAMTwin, kernels
from waam_twin.runtime import init_taichi
from waam_twin.solvers.coupled_step import _free_surface_stream

N, NZ, KF = 40, 22, 2
VOLUME = 718.0  # cells (hemisphere R = 7)
TAU = 0.6
GAMMA = 0.01
STEPS = 1500


def _cap_R(theta_deg: float) -> float:
    c = math.cos(math.radians(theta_deg))
    return (3.0 * VOLUME / (math.pi * (2.0 + c) * (1.0 - c) ** 2)) ** (1.0 / 3.0)


def _drop(theta: float, theta_init: float) -> tuple[float, float]:
    """Return (apparent contact angle, relative mass drift)."""
    twin = WAAMTwin(nx=N, ny=N, nz=NZ, dx=3e-4, max_tracers=10)
    twin.reset(T_ambient=float(twin.mat.T_liquidus) + 200.0, test_fluid_domain=True)
    g = twin.grid
    g.ensure_fslbm_fields()
    twin.enable_wetting = True
    twin.theta_rad = math.radians(theta)
    twin.enable_csf_tension = True
    twin.gamma_lu = GAMMA
    twin.fs_contact_slip = True

    I, J, K = np.meshgrid(
        np.arange(N) + 0.5, np.arange(N) + 0.5, np.arange(NZ) + 0.5, indexing="ij",
    )
    R0 = _cap_R(theta_init)
    zc = KF - R0 * math.cos(math.radians(theta_init))
    d = np.sqrt((I - N / 2) ** 2 + (J - N / 2) ** 2 + (K - zc) ** 2)
    fill = np.clip(R0 + 0.5 - d, 0.0, 1.0) * (K > KF)
    solid = K < KF
    fluid = (fill >= 0.999) & ~solid
    iface = (fill > 0.0) & ~fluid & ~solid
    flags = np.full(fill.shape, g.FLAG_GAS, np.int32)
    flags[solid] = g.FLAG_SOLID
    flags[fluid] = g.FLAG_FLUID
    flags[iface] = g.FLAG_IFACE
    g.flags.from_numpy(flags)
    g.phi.from_numpy(np.where(solid | fluid, 1.0, fill).astype(np.float32))
    g.f_l.from_numpy(np.ones(fill.shape, np.float32))

    def state() -> tuple[float, float, float]:
        f = g.flags.to_numpy()
        p = g.phi.to_numpy().astype(np.float64)
        rho = g.rho.to_numpy().astype(np.float64)
        liq = (f != g.FLAG_GAS) & (f != g.FLAG_SOLID)
        m = np.where(liq, p, 0.0)
        r_base = math.sqrt(float(m[:, :, KF].sum()) / math.pi)
        h = float(m[N // 2, N // 2, :].sum())
        return r_base, h, float((p * rho)[liq].sum())

    _r, _h, m0 = state()
    for _ in range(STEPS):
        kernels.clear_forces(g.Fx, g.Fy, g.Fz)
        kernels.collide_srt(
            g.f_src, g.f_dst, g.rho, g.ux, g.uy, g.uz, g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags, TAU, 1.0 / TAU, 1.0, 0.0,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        _free_surface_stream(twin, g)
    r, h, m1 = state()
    apparent = 2.0 * math.degrees(math.atan(h / max(r, 1e-9)))
    return apparent, abs(m1 - m0) / m0


def run() -> float:
    init_taichi(backend="cpu")
    worst_mass = 0.0

    a90, dm = _drop(90.0, 90.0)
    worst_mass = max(worst_mass, dm)
    print(f"[fslbm_wetting] θ=90°  apparent {a90:.1f}°  mass drift {dm:.1e}")
    if abs(a90 - 90.0) > 4.0:
        raise AssertionError(f"θ=90° hemisphere drifted to {a90:.1f}°")

    for theta, flat, steep in ((45.0, 30.0, 90.0), (135.0, 90.0, 160.0)):
        a_lo, dm1 = _drop(theta, flat)
        a_hi, dm2 = _drop(theta, steep)
        worst_mass = max(worst_mass, dm1, dm2)
        print(
            f"[fslbm_wetting] θ={theta:.0f}°  from {flat:.0f}° → {a_lo:.1f}°, "
            f"from {steep:.0f}° → {a_hi:.1f}°  (band {a_hi - a_lo:.1f}°)  "
            f"mass drift {max(dm1, dm2):.1e}"
        )
        if not (a_lo - 2.0 <= theta <= a_hi + 2.0):
            raise AssertionError(
                f"θ={theta}° not bracketed by the end states {a_lo:.1f}°–{a_hi:.1f}°"
            )
        if a_hi - a_lo > 20.0:
            raise AssertionError(
                f"θ={theta}° hysteresis band {a_hi - a_lo:.1f}° > 20° (contact line pinned)"
            )
    if worst_mass > 1e-3:
        raise AssertionError(f"metal mass drift {worst_mass:.1e} > 1e-3")
    print("[fslbm_wetting] PASS")
    return worst_mass


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
