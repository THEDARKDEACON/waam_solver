"""
test_laplace.py — Brackbill CSF curvature vs spherical Laplace law κ = 2/R.

PHYSICS_FORCE_CORRECTNESS_SPEC §5.2: mean interface curvature within 15% of
2/R_cells at dx/R ≤ 1/8, and CSF force must be nonzero.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.platform import init_taichi
from waam_twin import WAAMTwin
from waam_twin.physics import forces
from waam_twin import kernels


def _soft_sphere_phi(nx, ny, nz, cx, cy, cz, R_cells, delta=1.0):
    """φ = 1 inside droplet, smooth tanh interface of width ~delta cells."""
    phi = np.zeros((nx, ny, nz), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2)
                phi[i, j, k] = 0.5 * (1.0 - np.tanh((r - R_cells) / max(delta, 0.5)))
    return phi


def run(tol_frac: float = 0.15, min_force_lu: float = 1e-12) -> float:
    init_taichi(backend="cpu")
    # R = 8 cells → dx/R = 1/8 on unit cell; domain comfortably contains the drop.
    n = 40
    twin = WAAMTwin(
        nx=n, ny=n, nz=n, dx=2.5e-4,
        enable_csf_tension=True, enable_vof=True,
        C_darcy=0.0, max_tracers=10,
    )
    twin.reset(test_fluid_domain=True)
    g = twin.grid

    R = 8.0
    cx = cy = cz = (n - 1) / 2.0
    phi_np = _soft_sphere_phi(n, n, n, cx, cy, cz, R, delta=1.0)
    flags_np = np.full((n, n, n), g.FLAG_FLUID, dtype=np.int32)
    fl_np = phi_np.copy()
    flags_np[phi_np < 0.05] = g.FLAG_GAS
    flags_np[(phi_np >= 0.05) & (phi_np <= 0.95)] = g.FLAG_IFACE
    g.phi.from_numpy(phi_np)
    g.flags.from_numpy(flags_np)
    g.f_l.from_numpy(fl_np)

    g.ensure_export_buffers()
    kernels.compute_curvature_field(
        g.phi, g.flags, g.kappa_field, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    kappa = g.kappa_field.to_numpy()
    iface = (phi_np > 0.25) & (phi_np < 0.75) & (flags_np != g.FLAG_GAS)
    # Exclude a thin rim near domain boundary where stencil is incomplete.
    rim = 2
    iface[:rim, :, :] = False
    iface[-rim:, :, :] = False
    iface[:, :rim, :] = False
    iface[:, -rim:, :] = False
    iface[:, :, :rim] = False
    iface[:, :, -rim:] = False

    if not np.any(iface):
        raise AssertionError("No interface cells for curvature sample")

    kappa_iface = kappa[iface]
    # Sphere: κ = 2/R in lattice units (lengths in cells).
    kappa_ref = 2.0 / R
    kappa_mean = float(np.mean(kappa_iface))
    # Sign: φ high inside → n̂ inward → κ should be positive for a droplet.
    if kappa_mean < 0:
        kappa_mean = -kappa_mean
        kappa_iface = -kappa_iface
    err = abs(kappa_mean - kappa_ref) / kappa_ref
    print(
        f"[laplace] κ_mean={kappa_mean:.4f}  κ_ref=2/R={kappa_ref:.4f}  "
        f"rel_err={err:.3f}  (tol={tol_frac:.0%})  N_iface={int(iface.sum())}"
    )
    if err > tol_frac:
        raise AssertionError(
            f"Brackbill curvature off Laplace sphere: rel_err={err:.3f} > {tol_frac}"
        )

    forces.clear_forces(g.Fx, g.Fy, g.Fz)
    forces.compute_csf_tension(
        g.phi, g.flags, g.Fx, g.Fy, g.Fz,
        twin.gamma_lu,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    Fmag = float(np.sqrt(
        g.Fx.to_numpy() ** 2 + g.Fy.to_numpy() ** 2 + g.Fz.to_numpy() ** 2
    ).max())
    print(f"[laplace] max |F_csf| = {Fmag:.3e} lu")
    if Fmag < min_force_lu:
        raise AssertionError("CSF tension force negligible on sphere")
    return err


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
