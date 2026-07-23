"""
test_goldak_energy.py — Goldak inject deposits η Q Δt (±1%) and ff+fr→2.

PHYSICS_FORCE_CORRECTNESS_SPEC §5.4 / WP-F.
"""

from __future__ import annotations

import sys

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics.arc import Goldak3D, normalize_goldak_fractions


def run(tol_frac: float = 0.01) -> float:
    init_taichi(backend="cpu")

    ff, fr = normalize_goldak_fractions(0.6, 0.4)
    assert abs(ff + fr - 2.0) < 1e-9, f"normalize failed: ff+fr={ff + fr}"
    ff2, fr2 = normalize_goldak_fractions(0.6, 1.4)
    assert abs(ff2 - 0.6) < 1e-12 and abs(fr2 - 1.4) < 1e-12

    twin = WAAMTwin(
        nx=32, ny=24, nz=20, dx=3e-4,
        heat_source="goldak",
        arc_efficiency=0.8,
        arc_power_W=3600.0,  # 180 A × 20 V
        enable_vof=False,
        max_tracers=10,
    )
    twin.reset(test_fluid_domain=True)
    twin.arc_source = Goldak3D(
        ff=0.6, fr=1.4,
        a_front=4.0, a_rear=8.0, b=4.0, c=3.0,
    )
    g = twin.grid

    H0 = float(g.H.to_numpy().sum()) * (g.dx ** 3)
    arc_i = (g.nx - 1) * 0.5
    arc_j = (g.ny - 1) * 0.5
    # Place arc on the mid-plane metal so both front and rear halves exist.
    arc_k = float(twin.nz_solid + 2)
    twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)
    H1 = float(g.H.to_numpy().sum()) * (g.dx ** 3)

    dH = H1 - H0
    expected = twin.eta * twin.Q_w * g.dt
    err = abs(dH - expected) / max(expected, 1e-30)
    print(
        f"[goldak_energy] ΔH={dH:.6e} J  ηQΔt={expected:.6e} J  "
        f"rel_err={err:.4f}  (tol={tol_frac:.0%})"
    )
    if err > tol_frac:
        raise AssertionError(
            f"Goldak energy not conserved: |ΔH - ηQΔt|/ηQΔt = {err:.4f} > {tol_frac}"
        )

    # Front/rear asymmetry smoke: longer rear + larger fr ⇒ more heat behind torch.
    H_np = g.H.to_numpy()
    i0 = int(round(arc_i))
    front = float(H_np[i0 + 1 :, :, :].sum())
    rear = float(H_np[:i0, :, :].sum())
    print(f"[goldak_energy] H_front_cells={front:.3e}  H_rear_cells={rear:.3e}")
    if rear <= front:
        raise AssertionError(
            "Expected more enthalpy behind torch (fr>ff, a_rear>a_front)"
        )
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
