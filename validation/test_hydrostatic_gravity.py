"""
test_hydrostatic_gravity.py — gravity body force on liquid cells, any welding position.

Flat: weight pulls liquid toward the plate (−z) and Boussinesq buoyancy lifts
hot liquid (+z). Overhead flips both; vertical-down points them along the
travel direction (job.welding_position).
"""

from __future__ import annotations

import math

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi
from waam_twin.physics import forces


def _forces(twin, gdir):
    g = twin.grid
    i, j, k = g.nx // 2, g.ny // 2, twin.nz_solid + 2
    g.f_l[i, j, k] = 1.0
    g.flags[i, j, k] = g.FLAG_FLUID
    g.T[i, j, k] = twin.mat.T_liquidus + 300.0
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.add_hydrostatic_gravity(
        g.Fx, g.Fy, g.Fz, g.f_l, g.flags, 1.0, twin.g_lu, *gdir,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    w = (float(g.Fx[i, j, k]), float(g.Fy[i, j, k]), float(g.Fz[i, j, k]))
    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.add_buoyancy(
        g.T, g.Fx, g.Fy, g.Fz, g.f_l, g.flags, twin.g_lu, *gdir, twin.beta_T,
        twin.mat.T_liquidus, 1.0, g.FLAG_SOLID, g.FLAG_GAS,
    )
    b = (float(g.Fx[i, j, k]), float(g.Fy[i, j, k]), float(g.Fz[i, j, k]))
    return w, b


def _position_from_job() -> None:
    from waam_twin.job import gravity_direction_for
    assert gravity_direction_for("flat", (1.0, 0.0)) == (0.0, 0.0, -1.0)
    assert gravity_direction_for("overhead", (1.0, 0.0)) == (0.0, 0.0, 1.0)
    gx, gy, gz = gravity_direction_for("vertical_down", (0.0, 1.0))
    assert abs(gx) < 1e-12 and abs(gy - 1.0) < 1e-12 and gz == 0.0
    gx, gy, gz = gravity_direction_for("vertical_up", (1.0, 0.0))
    assert abs(gx + 1.0) < 1e-12 and gy == 0.0 and gz == 0.0


def run() -> None:
    init_taichi(backend="cpu")
    twin = WAAMTwin(nx=24, ny=16, nz=18, dx=3e-4, enable_hydrostatic_gravity=True)
    twin.reset()
    assert twin.gravity_dir == (0.0, 0.0, -1.0)
    w, b = _forces(twin, (0.0, 0.0, -1.0))
    assert w[2] < 0.0 and abs(w[0]) == 0.0 and abs(w[1]) == 0.0, w
    assert abs(w[2] + twin.g_lu) < 1e-6 * twin.g_lu, (w, twin.g_lu)
    assert b[2] > 0.0, "hot liquid must rise against gravity"
    w_o, b_o = _forces(twin, (0.0, 0.0, 1.0))
    assert math.isclose(w_o[2], -w[2], rel_tol=1e-6) and math.isclose(b_o[2], -b[2], rel_tol=1e-6)
    w_v, b_v = _forces(twin, (1.0, 0.0, 0.0))
    assert math.isclose(w_v[0], -w[2], rel_tol=1e-6) and w_v[2] == 0.0
    assert b_v[0] < 0.0 and b_v[2] == 0.0
    _position_from_job()
    print(f"[hydrostatic] flat Fz={w[2]:.3e} lu, buoyancy +{b[2]:.3e}; overhead and vertical flip/rotate")


if __name__ == "__main__":
    run()
    print("PASS")
