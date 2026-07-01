"""
test_wetting_static_settle.py — Lower contact angle → wider sessile spread after settle.

Runs a static molten puddle (no arc) with CSF + hydrostatic gravity + wall wetting.
"""

from __future__ import annotations

import numpy as np

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi


def _cap_metrics(twin, g) -> tuple[float, float]:
    """Return (half_width_mm, height_mm) of liquid above substrate."""
    phi = g.phi.to_numpy()
    nz = twin.nz_solid
    k0 = max(0, nz)
    mask_base = phi[:, :, k0] > 0.35
    if not mask_base.any():
        return 0.0, 0.0
    xs = np.where(mask_base.any(axis=1))[0]
    ys = np.where(mask_base.any(axis=0))[0]
    half_w = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * g.dx * 1000.0
    k_top = k0
    for k in range(k0, min(g.nz, k0 + 14)):
        if phi[:, :, k].max() > 0.45:
            k_top = k
    height = (k_top - k0) * g.dx * 1000.0
    return half_w, height


def _run_settle(theta_deg: float, n_steps: int = 500) -> tuple[float, float]:
    twin = WAAMTwin(
        nx=36, ny=28, nz=26, dx=2.5e-4,
        enable_vof=True,
        enable_csf_tension=True,
        enable_wetting=True,
        enable_hydrostatic_gravity=True,
        contact_angle_deg=theta_deg,
        max_tracers=10,
    )
    twin.reset()
    g = twin.grid
    nz_s = twin.nz_solid
    ci, cj = g.nx // 2, g.ny // 2
    phi_np = g.phi.to_numpy()
    flags_np = g.flags.to_numpy()
    T_np = g.T.to_numpy()
    fl_np = g.f_l.to_numpy()
    Tm = twin.mat.T_liquidus + 200.0
    for di in range(-4, 5):
        for dj in range(-4, 5):
            i, j = ci + di, cj + dj
            if 0 <= i < g.nx and 0 <= j < g.ny:
                k = nz_s
                if k < g.nz:
                    phi_np[i, j, k] = 1.0
                    flags_np[i, j, k] = g.FLAG_FLUID
                    T_np[i, j, k] = Tm
                    fl_np[i, j, k] = 1.0
                    if k + 1 < g.nz:
                        phi_np[i, j, k + 1] = 0.6
                        flags_np[i, j, k + 1] = g.FLAG_FLUID
                        T_np[i, j, k + 1] = Tm
                        fl_np[i, j, k + 1] = 1.0
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)
    g.T.from_numpy(T_np)
    g.f_l.from_numpy(fl_np)

    cy = cj * g.dx
    for _ in range(n_steps):
        twin.step(ci * g.dx, cy, is_welding=False)

    return _cap_metrics(twin, g)


def run() -> None:
    init_taichi(backend="cpu")
    w_narrow, h_narrow = _run_settle(85.0)
    w_wide, h_wide = _run_settle(55.0)
    ar_narrow = h_narrow / max(w_narrow, 0.1)
    ar_wide = h_wide / max(w_wide, 0.1)
    print(
        f"[wetting_static_settle] θ=85° w={w_narrow:.3f} h={h_narrow:.3f} mm h/w={ar_narrow:.2f}  "
        f"θ=55° w={w_wide:.3f} h={h_wide:.3f} mm h/w={ar_wide:.2f}"
    )
    if not (w_wide > w_narrow * 1.03 or ar_narrow > ar_wide * 1.08):
        raise AssertionError(
            f"lower θ should spread wider or stand taller: "
            f"85° w={w_narrow:.3f} h={h_narrow:.3f} vs 55° w={w_wide:.3f} h={h_wide:.3f}"
        )


if __name__ == "__main__":
    run()
    print("PASS")
