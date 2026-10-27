"""
test_viewer_surface.py — φ = 0.5 sheet covers the plate face and the bead crown.
"""

from __future__ import annotations

import numpy as np

from waam_twin.viewer.surface import metal_gas_k_band, phi_isosurface


def run() -> None:
    nx, ny, nz = 40, 24, 48
    dx_mm = 0.4
    nz_solid = 30
    phi = np.zeros((nx, ny, nz), np.float32)
    temperature = np.full((nx, ny, nz), 300.0, np.float32)
    phi[:, :, :nz_solid] = 1.0
    # 2 mm crown (5 cells) on the middle of the coupon.
    phi[12:28, 8:16, nz_solid:nz_solid + 5] = 1.0
    temperature[12:28, 8:16, nz_solid:nz_solid + 5] = 1800.0

    k0, k1 = metal_gas_k_band(phi, nz_solid, dx_mm)
    face = nz_solid * dx_mm
    assert k0 < nz_solid <= k1
    assert k0 > 0, "band walked the bottom of the coupon"
    assert k1 > nz_solid + 5, "band stopped short of the crown"
    assert (nz_solid - k0) * dx_mm <= 6.0 + dx_mm

    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("[viewer_surface] k-band ok; pyvista missing, contour skipped")
        return

    surf = phi_isosurface(
        phi,
        temperature,
        dx_mm=dx_mm,
        nz_solid=nz_solid,
        t_lo=300.0,
        t_hi=1800.0 + 800.0,
    )
    assert surf is not None and surf.n_cells > 0
    z = np.asarray(surf.points)[:, 2]
    assert z.min() >= (k0 - 0.51) * dx_mm
    assert z.max() > face + 1.0, f"crown missing, zmax={z.max():.2f} face={face:.2f}"
    assert z.min() < face + dx_mm, "plate face missing from the sheet"
    rgb = np.asarray(surf.point_data["RGB"])
    assert rgb.shape == (surf.n_points, 3)
    hot = rgb[z > face + 0.5, 0].max()
    cold = rgb[z < face + 0.2, 0].mean()
    assert hot > cold
    print(
        f"[viewer_surface] tris={surf.n_cells}  "
        f"k={k0}:{k1}  z={z.min():.2f}..{z.max():.2f} mm  face={face:.2f}"
    )


if __name__ == "__main__":
    run()
    print("PASS")
