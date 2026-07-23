"""VTK / PyVista export for WAAM research workflows."""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .. import kernels

if TYPE_CHECKING:
    from ..twin import WAAMTwin

TIER_CORE = 0
TIER_MATERIAL = 1
TIER_FORCES = 2
TIER_DERIVED = 3

TIER_MAP = {
    0: TIER_CORE,
    1: TIER_MATERIAL,
    2: TIER_FORCES,
    3: TIER_DERIVED,
}


def _headless() -> bool:
    return os.environ.get("WAAM_HEADLESS") == "1"


def _require_pyvista():
    try:
        import pyvista as pv  # noqa: F401
    except ImportError:
        return None
    import pyvista as pv
    return pv


def vtk_imagedata_path(path: str) -> str:
    p = pathlib.Path(path)
    if p.suffix.lower() in (".vts", ""):
        return str(p.with_suffix(".vti"))
    return path


def _image_grid(twin: "WAAMTwin"):
    pv = _require_pyvista()
    if pv is None:
        return None, None
    g = twin.grid
    grid_pv = pv.ImageData()
    grid_pv.dimensions = (g.nx + 1, g.ny + 1, g.nz + 1)
    grid_pv.spacing = (g.dx * 1000.0,) * 3
    grid_pv.origin = (twin._window_offset_x_m * 1000.0, 0.0, 0.0)
    return pv, grid_pv


def _lu_to_ms(g, arr: np.ndarray) -> np.ndarray:
    return arr * g.dx / g.dt


def _prepare_derived(twin: "WAAMTwin", tiers: Sequence[int]) -> None:
    if TIER_DERIVED not in tiers:
        return
    g = twin.grid
    g.ensure_export_buffers()
    kernels.compute_curvature_field(
        g.phi, g.flags, g.kappa_field,
        g.FLAG_GAS, g.nx, g.ny, g.nz,
    )
    kernels.compute_vorticity_magnitude(
        g.ux, g.uy, g.uz, g.flags, g.vorticity_mag,
        g.dx, g.dt, g.FLAG_GAS, g.nx, g.ny, g.nz,
    )


def export_volume(
    twin: "WAAMTwin",
    path: str,
    tiers: Sequence[int] = (TIER_CORE, TIER_DERIVED),
    crop_liquid: bool = False,
) -> str | None:
    if _headless():
        return None
    pv = _require_pyvista()
    if pv is None:
        print("[export] pyvista not installed. Skipping volume VTK.")
        return None

    g = twin.grid
    _prepare_derived(twin, tiers)

    _, grid_pv = _image_grid(twin)
    assert grid_pv is not None

    T_np = g.T.to_numpy()
    fl_np = g.f_l.to_numpy()
    ux_np = g.ux.to_numpy()
    uy_np = g.uy.to_numpy()
    uz_np = g.uz.to_numpy()
    ux_ms = _lu_to_ms(g, ux_np)
    uy_ms = _lu_to_ms(g, uy_np)
    uz_ms = _lu_to_ms(g, uz_np)
    speed = np.sqrt(ux_ms ** 2 + uy_ms ** 2 + uz_ms ** 2)

    if TIER_CORE in tiers:
        grid_pv.cell_data["Temperature_K"] = T_np.ravel(order="F")
        grid_pv.cell_data["Temperature_C"] = (T_np - 273.15).ravel(order="F")
        grid_pv.cell_data["Liquid_Fraction"] = fl_np.ravel(order="F")
        grid_pv.cell_data["VOF_phi"] = g.phi.to_numpy().ravel(order="F")
        grid_pv.cell_data["Cell_Flags"] = g.flags.to_numpy().ravel(order="F")
        grid_pv.cell_data["T_max_K"] = g.T_max.to_numpy().ravel(order="F")
        grid_pv.cell_data["T_prev_K"] = g.T_prev.to_numpy().ravel(order="F")
        # dT_dt is (T − T_prev)/dt, i.e. positive while HEATING. The cooling
        # rate (positive while cooling, the metallurgically meaningful sign)
        # is its negation — previously the raw signal was exported under the
        # cooling-rate name, flipping the sign of every reported CCT value.
        dT_dt_np = g.dT_dt.to_numpy()
        grid_pv.cell_data["dTdt_Ks"] = dT_dt_np.ravel(order="F")
        grid_pv.cell_data["Cooling_Rate_Ks"] = (-dT_dt_np).ravel(order="F")
        grid_pv.cell_data["Enthalpy_Jm3"] = g.H.to_numpy().ravel(order="F")
        grid_pv.cell_data["Time_above_800C_s"] = g.time_above_800_s.to_numpy().ravel(order="F")
        grid_pv.cell_data["Time_above_1100C_s"] = g.time_above_1100_s.to_numpy().ravel(order="F")
        grid_pv.cell_data["Time_above_solidus_s"] = g.time_above_solidus_s.to_numpy().ravel(order="F")
        grid_pv.cell_data["Velocity_X_ms"] = ux_ms.ravel(order="F")
        grid_pv.cell_data["Velocity_Y_ms"] = uy_ms.ravel(order="F")
        grid_pv.cell_data["Velocity_Z_ms"] = uz_ms.ravel(order="F")
        grid_pv.cell_data["Speed_ms"] = speed.ravel(order="F")
        grid_pv.cell_data["Density_lu"] = g.rho.to_numpy().ravel(order="F")
        cs2 = 1.0 / 3.0
        # Gauge pressure: p = cs²·(ρ_lu − 1)·ρ_phys·(dx/dt)². Using the raw
        # ρ_lu gave a huge constant offset that swamped the dynamic signal.
        p_pa = (g.rho.to_numpy() - 1.0) * cs2 * g.mat.rho * (g.dx / g.dt) ** 2
        grid_pv.cell_data["Pressure_Pa_gauge"] = p_pa.ravel(order="F")

    if TIER_MATERIAL in tiers and twin.use_material_tables:
        grid_pv.cell_data["RhoCp_Jm3K"] = g.cp_rho_field.to_numpy().ravel(order="F")
        grid_pv.cell_data["Alpha_lu"] = g.alpha_lu_field.to_numpy().ravel(order="F")
        grid_pv.cell_data["DgammaDT_lu"] = g.dgamma_lu_field.to_numpy().ravel(order="F")
        grid_pv.cell_data["Tau_SRT"] = g.tau_field.to_numpy().ravel(order="F")

    if TIER_FORCES in tiers:
        fx = _lu_to_ms(g, g.Fx_snap.to_numpy()) / g.dt
        fy = _lu_to_ms(g, g.Fy_snap.to_numpy()) / g.dt
        fz = _lu_to_ms(g, g.Fz_snap.to_numpy()) / g.dt
        grid_pv.cell_data["BodyForce_X_ms2"] = fx.ravel(order="F")
        grid_pv.cell_data["BodyForce_Y_ms2"] = fy.ravel(order="F")
        grid_pv.cell_data["BodyForce_Z_ms2"] = fz.ravel(order="F")
        fmag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        grid_pv.cell_data["BodyForce_mag_ms2"] = fmag.ravel(order="F")
        if twin.enable_lorentz:
            g.ensure_lorentz_fields()
            grid_pv.cell_data["CurrentDensity_X_Am2"] = g.Jx.to_numpy().ravel(order="F")
            grid_pv.cell_data["CurrentDensity_Y_Am2"] = g.Jy.to_numpy().ravel(order="F")
            grid_pv.cell_data["CurrentDensity_Z_Am2"] = g.Jz.to_numpy().ravel(order="F")
            grid_pv.cell_data["MagneticField_X_T"] = g.Bx.to_numpy().ravel(order="F")
            grid_pv.cell_data["MagneticField_Y_T"] = g.By.to_numpy().ravel(order="F")
            grid_pv.cell_data["MagneticField_Z_T"] = g.Bz.to_numpy().ravel(order="F")

    if TIER_DERIVED in tiers:
        grid_pv.cell_data["Curvature_kappa"] = g.kappa_field.to_numpy().ravel(order="F")
        grid_pv.cell_data["Vorticity_mag_1s"] = g.vorticity_mag.to_numpy().ravel(order="F")
        grid_pv.cell_data["Mushy_Zone"] = (
            (fl_np > 0.05) & (fl_np < 0.95)
        ).astype(np.float32).ravel(order="F")

    if crop_liquid and "Liquid_Fraction" in grid_pv.cell_data:
        grid_pv = grid_pv.threshold(0.05, scalars="Liquid_Fraction")

    out = vtk_imagedata_path(path)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    grid_pv.save(out)
    print(f"[export] Volume VTK → {out}  ({len(grid_pv.cell_data)} arrays)")
    return out


def export_surface(
    twin: "WAAMTwin",
    path: str,
    include_kappa: bool = True,
) -> str | None:
    if _headless():
        return None
    pv = _require_pyvista()
    if pv is None:
        print("[export] pyvista not installed. Skipping surface VTK.")
        return None

    g = twin.grid
    if include_kappa:
        g.ensure_export_buffers()
        kernels.compute_curvature_field(
            g.phi, g.flags, g.kappa_field,
            g.FLAG_GAS, g.nx, g.ny, g.nz,
        )

    _, grid_pv = _image_grid(twin)
    assert grid_pv is not None

    phi_np = g.phi.to_numpy()
    fl_np = g.f_l.to_numpy()
    grid_pv.cell_data["phi"] = phi_np.ravel(order="F")
    grid_pv.cell_data["Liquid_Fraction"] = fl_np.ravel(order="F")
    grid_pv.cell_data["Temperature_K"] = g.T.to_numpy().ravel(order="F")
    if include_kappa:
        grid_pv.cell_data["Curvature_kappa"] = g.kappa_field.to_numpy().ravel(order="F")

    point_grid = grid_pv.cell_data_to_point_data()
    surf = None
    # φ=0.5 is the metal/gas free surface (the bead profile). Liquid_Fraction
    # =0.5 is the melt front — only used as fallback when VOF is disabled and
    # phi carries no interface. (Previously f_l was tried first, so the
    # "bead_surface" export was silently the melt front.)
    for scalar in ("phi", "Liquid_Fraction"):
        try:
            candidate = point_grid.contour([0.5], scalars=scalar)
        except Exception:
            candidate = point_grid.contour(isosurfaces=[0.5], scalars=scalar)
        if candidate.n_cells > 0:
            surf = candidate
            break
    if surf is None or surf.n_cells == 0:
        print("[export] No φ/f_l=0.5 surface found; skipping surface VTK.")
        return None

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    surf.save(path)
    print(f"[export] Surface VTK → {path}  ({surf.n_cells} cells)")
    return path


def export_tracers(twin: "WAAMTwin", path: str) -> str | None:
    if _headless():
        return None
    pv = _require_pyvista()
    if pv is None:
        return None

    g = twin.grid
    pos = g.porosity_pos.to_numpy()
    active = g.porosity_active.to_numpy()
    mask = active > 0
    if not mask.any():
        print("[export] No active tracers; skipping tracer VTK.")
        return None

    ox = twin._window_offset_x_m
    pts = pos[mask].copy()
    pts[:, 0] += ox
    pts *= 1000.0  # mm

    cloud = pv.PolyData(pts)
    cloud["state"] = active[mask]
    cloud["trapped"] = (active[mask] == 2).astype(np.int32)

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    cloud.save(path)
    print(f"[export] Tracers VTK → {path}  ({cloud.n_points} points)")
    return path


def export_legacy_minimal(twin: "WAAMTwin", path: str) -> str | None:
    """Backward-compatible slim export (original export_vtk fields + Velocity_Y)."""
    return export_volume(twin, path, tiers=(TIER_CORE,), crop_liquid=False)
