"""
test_goldak_energy.py — Goldak inject deposits η Q Δt (±1%) and ff+fr→2.

PHYSICS_FORCE_CORRECTNESS_SPEC §5.4 / WP-F.
Also checks travel-aligned front/rear for +x and +y torch directions.
"""

from __future__ import annotations

import sys

from waam_twin import WAAMTwin
from waam_twin.platform import init_taichi
from waam_twin.physics.arc import Goldak3D, normalize_goldak_fractions, create_heat_source


def _front_rear_enthalpy(H_np, arc_i: float, arc_j: float, dir_x: float, dir_y: float):
    """Sum H on front vs rear half-planes in the travel frame."""
    nx, ny, _nz = H_np.shape
    front = 0.0
    rear = 0.0
    for i in range(nx):
        for j in range(ny):
            di = float(i) - arc_i
            dj = float(j) - arc_j
            x_travel = di * dir_x + dj * dir_y
            col = float(H_np[i, j, :].sum())
            if x_travel >= 0.0:
                front += col
            else:
                rear += col
    return front, rear


def run(tol_frac: float = 0.01) -> float:
    init_taichi(backend="cpu")

    ff, fr = normalize_goldak_fractions(0.6, 0.4)
    assert abs(ff + fr - 2.0) < 1e-9, f"normalize failed: ff+fr={ff + fr}"
    ff2, fr2 = normalize_goldak_fractions(0.6, 1.4)
    assert abs(ff2 - 0.6) < 1e-12 and abs(fr2 - 1.4) < 1e-12

    try:
        create_heat_source("not_a_real_source")
        raise AssertionError("unknown heat_source should raise")
    except ValueError:
        pass
    try:
        create_heat_source("conical")
        raise AssertionError("conical heat_source should raise")
    except ValueError:
        pass

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
    arc_i = (g.nx - 1) * 0.5
    arc_j = (g.ny - 1) * 0.5
    # Place arc on the mid-plane metal so both front and rear halves exist.
    arc_k = float(twin.nz_solid + 2)

    # Zero ambient enthalpy so front/rear compares deposited energy only.
    g.H.fill(0.0)
    twin._torch_dir_xyz = (1.0, 0.0, 0.0)
    twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)
    H1 = float(g.H.to_numpy().sum()) * (g.dx ** 3)

    expected = twin.eta * twin.Q_w * g.dt
    err = abs(H1 - expected) / max(expected, 1e-30)
    print(
        f"[goldak_energy] ΔH={H1:.6e} J  ηQΔt={expected:.6e} J  "
        f"rel_err={err:.4f}  (tol={tol_frac:.0%})"
    )
    if err > tol_frac:
        raise AssertionError(
            f"Goldak energy not conserved: |ΔH - ηQΔt|/ηQΔt = {err:.4f} > {tol_frac}"
        )

    # Front/rear asymmetry: longer rear + larger fr ⇒ more heat behind torch (+x).
    H_np = g.H.to_numpy()
    front, rear = _front_rear_enthalpy(H_np, arc_i, arc_j, 1.0, 0.0)
    print(f"[goldak_energy] +x travel  H_front={front:.3e}  H_rear={rear:.3e}")
    if rear <= front:
        raise AssertionError(
            "Expected more enthalpy behind torch on +x (fr>ff, a_rear>a_front)"
        )

    # +y travel: clear and re-inject with torch direction along +y.
    g.H.fill(0.0)
    twin.arc_source = Goldak3D(
        ff=0.6, fr=1.4,
        a_front=4.0, a_rear=8.0, b=4.0, c=3.0,
    )
    twin._torch_dir_xyz = (0.0, 1.0, 0.0)
    twin.arc_source.inject(twin, g, arc_i, arc_j, arc_k)
    H_np = g.H.to_numpy()
    front_y, rear_y = _front_rear_enthalpy(H_np, arc_i, arc_j, 0.0, 1.0)
    print(f"[goldak_energy] +y travel  H_front={front_y:.3e}  H_rear={rear_y:.3e}")
    if rear_y <= front_y:
        raise AssertionError(
            "Expected more enthalpy behind torch on +y travel direction"
        )
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
