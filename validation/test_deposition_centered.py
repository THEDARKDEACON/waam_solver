"""
test_deposition_centered.py — droplet metal lands centred on the arc.

A droplet budget smaller than the footprint must be placed nearest the arc
first, so the deposited metal's centroid sits on the arc axis. Checked for
the gas-cell path (feed_wire_surface) and the interface top-up path
(fs_deposit_into_interface).

Regression: both kernels spent the budget serially in (i, j, k) index order
from the −x/−y corner of the search box, so a short budget was placed
off-centre toward −x/−y whatever the travel direction.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin import WAAMTwin
from waam_twin import kernels
from waam_twin.physics import deposition
from waam_twin.runtime import init_taichi

TOL_CELLS = 0.5


def _twin() -> WAAMTwin:
    twin = WAAMTwin(nx=40, ny=40, nz=20, dx=3e-4, enable_vof=True, max_tracers=10)
    twin.reset()
    return twin


def _centroid(mask: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    I, J, _K = np.meshgrid(*(np.arange(n) for n in mask.shape), indexing="ij")
    w = np.where(mask, weight, 0.0)
    tot = float(w.sum())
    if tot <= 0.0:
        raise AssertionError("no metal deposited")
    return float((w * I).sum() / tot), float((w * J).sum() / tot)


def _check(label: str, ci: float, cj: float, arc_i: float, arc_j: float) -> float:
    off = float(np.hypot(ci - arc_i, cj - arc_j))
    print(
        f"[deposition_centered] {label:<9} centroid=({ci:.2f},{cj:.2f}) "
        f"arc=({arc_i:.1f},{arc_j:.1f})  offset={off:.2f} cells (tol {TOL_CELLS})"
    )
    if off > TOL_CELLS:
        raise AssertionError(f"{label}: deposit centroid {off:.2f} cells off the arc axis")
    return off


def run() -> float:
    init_taichi(backend="cpu")
    arc_i, arc_j = 20.0, 20.0

    # Gas-cell path: flat plate, footprint 8 cells, budget ≈ 20 cells.
    twin = _twin()
    g = twin.grid
    k_arc = float(twin.nz_solid - 1)
    flags0 = g.flags.to_numpy()
    budget = 20.0 * g.dx ** 3
    g.deposit_vol_buf[None] = 0.0
    g.deposit_real_buf[None] = 0.0
    deposition.feed_wire_surface(
        g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
        arc_i, arc_j, k_arc, 8.0, 2.0, budget,
        float(twin.mat.T_liquidus) + 300.0, twin.cp_rho, twin.L_rho, 1.0,
        g.deposit_vol_buf, g.deposit_real_buf, g.dx ** 3,
        g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID, g.nx, g.ny, g.nz,
    )
    new = (g.flags.to_numpy() == g.FLAG_FLUID) & (flags0 == g.FLAG_GAS)
    ci, cj = _centroid(new, np.ones(new.shape))
    worst = _check("gas", ci, cj, arc_i, arc_j)

    # Interface path: a layer of half-full IFACE cells on the plate.
    twin = _twin()
    g = twin.grid
    g.ensure_fslbm_fields()
    k_if = twin.nz_solid
    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()
    flags[:, :, k_if] = g.FLAG_IFACE
    phi[:, :, k_if] = 0.5
    g.flags.from_numpy(flags)
    g.phi.from_numpy(phi)
    budget = 10.0 * g.dx ** 3  # 20 cells × 0.5 fill
    g.deposit_vol_buf[None] = 0.0
    g.deposit_real_buf[None] = 0.0
    kernels.fs_deposit_into_interface(
        g.phi, g.flags, g.H, arc_i, arc_j, float(k_if - 1), 8.0, 2.0, budget,
        1.0e10, g.deposit_vol_buf, g.deposit_real_buf, g.dx ** 3,
        g.FLAG_IFACE, g.nx, g.ny, g.nz,
    )
    gained = g.phi.to_numpy() - phi
    ci, cj = _centroid(gained > 1e-6, gained)
    worst = max(worst, _check("interface", ci, cj, arc_i, arc_j))
    print("[deposition_centered] PASS")
    return worst


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
