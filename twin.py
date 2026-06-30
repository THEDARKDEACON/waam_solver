"""
twin.py — WAAMTwin v2 orchestrator
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from .grid import WAAMGrid
from .materials import MaterialProps, load_material
from .gpu_tables import MaterialGPUTables
from .physics.arc import create_heat_source
from . import kernels
from .solvers.coupled_step import coupled_step


class WAAMTwin:
    """GPU-resident WAAM melt-pool digital twin (Taichi LBM + enthalpy-porosity)."""

    def __init__(
        self,
        material: str | MaterialProps = "ER70S-6",
        nx: int = 256,
        ny: int = 128,
        nz: int = 64,
        dx: float = 2e-4,
        arc_power_W: float = 2800.0,
        arc_efficiency: float = 0.8,
        high_sulphur: bool = False,
        T_ambient: float = 300.0,
        use_srt: bool = True,
        C_darcy: float = 1.6e5,
        arc_pressure_pa: float = 500.0,
        droplet_freq_hz: float = 50.0,
        max_tracers: int | None = None,
        enable_heat_loss: bool = False,
        h_conv: float = 25.0,
        eps_rad: float = 0.3,
        enable_convection: bool = True,
        enable_radiation: bool = False,
        heat_source: str = "gaussian2d",
        enable_vof: bool = False,
        enable_recoil: bool = False,
        enable_csf_tension: bool = False,
        enable_deposition_momentum: bool = True,
        enable_substrate_growth: bool = False,
        enable_moving_window: bool = False,
        recoil_pressure_pa: float = 5000.0,
        travel_speed_m_s: float = 0.0,
        use_variable_tau: bool = True,
    ):
        if isinstance(material, MaterialProps):
            self.mat = material
        else:
            self.mat = load_material(material, high_sulphur)

        if max_tracers is None:
            max_tracers = 50_000

        self.grid = WAAMGrid(nx, ny, nz, dx, self.mat, max_tracers=max_tracers)
        self.Q_w = arc_power_W
        self.eta = arc_efficiency
        self.T_amb = T_ambient
        self.use_srt = use_srt
        self.C_darcy = C_darcy
        self.arc_pressure = arc_pressure_pa
        self.droplet_freq = droplet_freq_hz
        self.enable_heat_loss = enable_heat_loss
        self.h_conv = h_conv
        self.eps_rad = eps_rad
        self.enable_convection = enable_convection
        self.enable_radiation = enable_radiation
        self.sigma_sb = 5.670374419e-8
        self.heat_source_name = heat_source
        self.enable_vof = enable_vof
        self.enable_recoil = enable_recoil
        self.enable_csf_tension = enable_csf_tension
        self.enable_deposition_momentum = enable_deposition_momentum
        self.enable_substrate_growth = enable_substrate_growth
        self.enable_moving_window = enable_moving_window
        self.recoil_pressure_pa = recoil_pressure_pa
        self.travel_speed_m_s = travel_speed_m_s
        self.use_variable_tau = use_variable_tau
        self.nz_solid = max(1, nz // 5)
        self._step_n = 0
        self._last_droplet_time = 0.0
        self._window_offset_x_m = 0.0
        self._interpass_cooling_steps = 0
        self.preset_name: str | None = None

        kernels.bind_velocity_set(self.grid)

        g = self.grid
        mat = self.mat

        self.cp_rho = mat.rho * mat.cp
        self.L_rho = mat.rho * mat.L_fusion
        self.alpha_lu = mat.alpha * g.dt / (g.dx ** 2)
        self.omega = 1.0 / g.tau
        self.sigma_cells = 0.002 / g.dx
        self.g_lu = 9.81 * (g.dt ** 2) / g.dx
        self.beta_T = mat.beta_T

        scale_F = (g.dt ** 2) / (mat.rho * g.dx ** 2)
        self.force_scale = scale_F
        self.dgamma_dT_lu = mat.dgamma_dT * scale_F
        self.gamma_lu = mat.gamma_0 * scale_F / g.dx
        self.droplet_vz_lu = -0.03

        self.gpu_tables = MaterialGPUTables(mat)
        self.use_material_tables = self.gpu_tables.enabled
        cp_sol = mat.cp_at(mat.T_solidus)
        self.H_sol = mat.rho * cp_sol * mat.T_solidus
        self.H_liq = self.H_sol + self.L_rho
        self.arc_source = create_heat_source(heat_source)

        print(f"[WAAMTwin] Material: {mat.name}  status={mat.status}")
        if self.use_material_tables:
            parts = ["cp", "k", "dγ/dT"]
            if self.use_variable_tau and self.mat.tables.mu:
                parts.append("μ→τ")
            print(f"[WAAMTwin] T-dependent tables: {', '.join(parts)} on GPU")
        if self.enable_vof:
            print(f"[WAAMTwin] VOF advection enabled")
        print(f"[WAAMTwin] Heat source: {heat_source}")
        print(f"[WAAMTwin] VRAM estimate: {g.estimated_vram_mb():.1f} MB")
        print(f"[WAAMTwin] α_lu={self.alpha_lu:.6f}  τ_T={g.tau_T:.4f}")
        print(f"[WAAMTwin] Collision: {'SRT' if use_srt else 'MRT'}")

    @classmethod
    def from_preset(
        cls,
        preset: str = "standard",
        material: str = "ER70S-6",
        **kwargs: Any,
    ) -> "WAAMTwin":
        from .platform import (
            auto_grid,
            auto_tracer_count,
            check_vram_budget,
            ensure_taichi,
            resolve_preset,
        )

        ensure_taichi()
        cfg = resolve_preset(preset)
        vram = cfg.vram_budget_mb
        tracers = auto_tracer_count(vram, cfg)
        nx, ny, nz, dx = auto_grid(cfg.domain_mm, cfg.target_dx_mm, vram, tracers)
        check_vram_budget(nx, ny, nz, tracers, cfg.vram_budget_mb)

        twin = cls(
            material=material,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            use_srt=cfg.use_srt,
            max_tracers=tracers,
            **kwargs,
        )
        twin.preset_name = cfg.name
        print(f"[WAAMTwin] Preset={cfg.name}  grid={nx}×{ny}×{nz}  dx={dx*1e3:.3f}mm")
        return twin

    @classmethod
    def from_job(cls, job_path: str | pathlib.Path, **kwargs: Any) -> "WAAMTwin":
        from .job import load_job_config
        from .platform import ensure_taichi

        ensure_taichi()
        job = load_job_config(job_path)
        preset = job.get("simulation", {}).get("preset", "standard")
        material = job.get("material", "ER70S-6")
        process = job.get("process", {})

        arc_w = float(process.get("voltage_V", 20)) * float(process.get("current_A", 140))
        arc_eta = float(process.get("arc_efficiency", kwargs.pop("arc_efficiency", 0.8)))
        T_amb = float(process.get("T_ambient_K", kwargs.pop("T_ambient", 300.0)))

        heat = job.get("heat_loss", {})
        heat_kwargs = {}
        if heat:
            heat_kwargs = dict(
                enable_heat_loss=bool(heat.get("convection") or heat.get("radiation")),
                enable_convection=bool(heat.get("convection", False)),
                enable_radiation=bool(heat.get("radiation", False)),
                h_conv=float(heat.get("h_conv", 25.0)),
                eps_rad=float(heat.get("eps_rad", 0.3)),
            )

        twin = cls.from_preset(
            preset=preset,
            material=material,
            arc_power_W=arc_w,
            arc_efficiency=arc_eta,
            T_ambient=T_amb,
            heat_source=str(job.get("heat_source", kwargs.pop("heat_source", "gaussian2d"))),
            enable_vof=bool(job.get("simulation", {}).get("enable_vof", kwargs.pop("enable_vof", False))),
            **heat_kwargs,
            **kwargs,
        )
        from .job import apply_job_to_twin
        apply_job_to_twin(twin, job)
        twin._job_config = job
        return twin

    def reset(self, T_ambient: float | None = None, test_fluid_domain: bool = False):
        T0 = T_ambient or self.T_amb
        g = self.grid
        nz_solid = -1 if test_fluid_domain else max(1, g.nz // 5)
        self.nz_solid = max(1, g.nz // 5) if not test_fluid_domain else 0

        kernels.init_grid(
            g.f_a, g.rho, g.ux, g.uy, g.uz,
            g.T, g.H, g.f_l, g.phi, g.flags,
            T0, g.mat.rho, self.cp_rho,
            nz_solid,
            g.FLAG_FLUID, g.FLAG_SOLID, g.FLAG_GAS,
        )
        kernels.init_aux_fields(
            g.T_max, g.T_prev, g.dT_dt, g.Fx, g.Fy, g.Fz, T0,
            g.porosity_active, g.tracer_head, g.max_tracers,
        )
        kernels.stream(
            g.f_a, g.f_b, g.flags,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        self._step_n = 0
        self._last_droplet_time = 0.0
        print(f"[WAAMTwin] Grid reset. T_ambient={T0:.1f}K  test_fluid={test_fluid_domain}")

    def step(self, torch_x_m: float, torch_y_m: float, is_welding: bool = True):
        if self.enable_moving_window:
            self._maybe_shift_window(torch_x_m)
        sim_x = torch_x_m - self._window_offset_x_m
        sim_y = torch_y_m
        coupled_step(self, sim_x, sim_y, is_welding)

    def _maybe_shift_window(self, torch_x_world: float) -> None:
        g = self.grid
        sim_x = torch_x_world - self._window_offset_x_m
        trigger = 0.62 * (g.nx - 8) * g.dx
        if sim_x < trigger:
            return
        n_shift = max(8, g.nx // 10)
        from . import kernels

        kernels.shift_simulation_window_x(
            n_shift,
            g.f_a, g.f_b,
            g.T, g.H, g.f_l, g.phi, g.flags,
            g.T_max, g.T_prev, g.dT_dt,
            g.rho, g.ux, g.uy, g.uz, g.Fx, g.Fy, g.Fz,
            g.cp_rho_field, g.alpha_lu_field, g.dgamma_lu_field, g.tau_field,
            self.T_amb, self.cp_rho, g.mat.rho,
            self.nz_solid,
            g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_FLUID,
            g.nx, g.ny, g.nz,
        )
        self._window_offset_x_m += n_shift * g.dx
        pos = g.porosity_pos.to_numpy()
        pos[:, 0] -= n_shift * g.dx
        g.porosity_pos.from_numpy(pos)
        print(f"[WAAMTwin] Moving window shift −{n_shift} cells  offset={self._window_offset_x_m*1e3:.2f}mm")

    def run(self, n_steps: int, torch_x_m: float = 0.0,
            torch_y_m: float = 0.0, is_welding: bool = True):
        for _ in range(n_steps):
            self.step(torch_x_m, torch_y_m, is_welding)

    def run_path(
        self,
        job: dict | str | pathlib.Path,
        n_steps: int | None = None,
        is_welding: bool = True,
        interpass_steps: int = 0,
    ) -> None:
        """Follow torch_path waypoints in a job file at travel_speed_m_s."""
        from .job import load_job_config, parse_torch_path

        from .torch_path import TorchPathDriver, clamp_torch_to_domain

        if not isinstance(job, dict):
            job = load_job_config(job)
        waypoints = parse_torch_path(job)
        g = self.grid
        if not waypoints:
            cy = (g.ny // 2) * g.dx
            waypoints = [(0.01, cy, 0.0)]
        driver = TorchPathDriver(waypoints, self.travel_speed_m_s)

        if n_steps is None:
            n_steps = max(400, int(driver.total_length / max(self.travel_speed_m_s * g.dt, 1e-12)))

        interpass = int(getattr(self, "_interpass_cooling_steps", 0) or interpass_steps)
        prev_seg = -1

        for step, x, y, _z in driver.positions_for_steps(n_steps, g.dt):
            seg = driver.segment_index_at_distance(driver.distance_at_step(step, g.dt))
            if interpass > 0 and seg > prev_seg and prev_seg >= 0:
                for _ in range(interpass):
                    cx, cy = clamp_torch_to_domain(x, y, g.nx, g.ny, g.dx)
                    self.step(cx, cy, is_welding=False)
            prev_seg = seg
            cx, cy = clamp_torch_to_domain(x, y, g.nx, g.ny, g.dx)
            self.step(cx, cy, is_welding)

    def export_haz_vtk(self, path: str = "haz_map.vts") -> None:
        """Export peak temperature (HAZ) field to VTK."""
        if __import__("os").environ.get("WAAM_HEADLESS") == "1":
            return
        try:
            import pyvista as pv
        except ImportError:
            print("[WAAMTwin] pyvista not installed. Skipping HAZ VTK export.")
            return
        g = self.grid
        grid_pv = pv.ImageData()
        grid_pv.dimensions = (g.nx + 1, g.ny + 1, g.nz + 1)
        grid_pv.spacing = (g.dx * 1000,) * 3
        grid_pv.cell_data["T_max_K"] = g.T_max.to_numpy().ravel(order="F")
        grid_pv.cell_data["T_current_K"] = g.T.to_numpy().ravel(order="F")
        grid_pv.save(path)
        print(f"[WAAMTwin] HAZ VTK exported → {path}")

    def get_telemetry(self) -> dict:
        g = self.grid
        T_np = g.T.to_numpy()
        fl_np = g.f_l.to_numpy()
        dT_np = g.dT_dt.to_numpy()
        ux_np = g.ux.to_numpy()
        uz_np = g.uz.to_numpy()

        liquid_mask = fl_np > 0.5
        if self.preset_name == "minimal":
            j_mid = g.ny // 2
            liquid_mask &= np.arange(g.ny)[None, :, None] == j_mid

        n_liquid = int(np.count_nonzero(liquid_mask))

        if n_liquid > 0:
            T_peak = float(T_np[liquid_mask].max())
            peak_cool = float((-dT_np[liquid_mask]).max())
            x_idx = np.where(liquid_mask.any(axis=(1, 2)))[0]
            pool_width_m = (x_idx[-1] - x_idx[0]) * g.dx if len(x_idx) > 1 else 0.0
            z_idx = np.where(liquid_mask.any(axis=(0, 1)))[0]
            pool_depth_m = (z_idx[-1] - z_idx[0]) * g.dx if len(z_idx) > 1 else 0.0
            u_max_phys = float(
                np.sqrt(ux_np[liquid_mask] ** 2 + uz_np[liquid_mask] ** 2).max()
            ) * g.dx / g.dt
        else:
            T_peak = float(T_np.max())
            peak_cool = float((-dT_np).max())
            pool_width_m = pool_depth_m = u_max_phys = 0.0

        active = g.porosity_active.to_numpy()
        n_trapped = int((active == 2).sum())
        n_active = int((active == 1).sum())
        porosity_pct = 100.0 * n_trapped / max(n_trapped + n_active, 1)

        return {
            "step": self._step_n,
            "sim_time_ms": round(self._step_n * g.dt * 1000, 4),
            "peak_temp_K": round(T_peak, 1),
            "peak_temp_C": round(T_peak - 273.15, 1),
            "peak_cooling_rate_Ks": round(peak_cool, 1),
            "pool_width_mm": round(pool_width_m * 1000, 3),
            "pool_depth_mm": round(pool_depth_m * 1000, 3),
            "n_liquid_cells": n_liquid,
            "marangoni_vel_ms": round(u_max_phys, 4),
            "material_status": self.mat.status,
            "material_name": self.mat.name,
            "heat_source": self.heat_source_name,
            "material_tables": self.use_material_tables,
            "vof_enabled": self.enable_vof,
            "travel_speed_mm_s": round(self.travel_speed_m_s * 1000, 3),
            "variable_tau": self.use_variable_tau and self.use_material_tables,
            "porosity_pct": round(porosity_pct, 3),
            "n_trapped_tracers": n_trapped,
            "window_offset_x_mm": round(self._window_offset_x_m * 1000, 3),
        }

    def export_vtk(self, path: str = "weld_pool.vts"):
        if __import__("os").environ.get("WAAM_HEADLESS") == "1":
            return
        try:
            import shutil
            usage = shutil.disk_usage(pathlib.Path(path).parent)
            if usage.free < 50 * 1024 * 1024:
                print("[WAAMTwin] Low disk space (<50 MB). Skipping VTK export.")
                return
        except OSError:
            pass
        try:
            import pyvista as pv
        except ImportError:
            print("[WAAMTwin] pyvista not installed. Skipping VTK export.")
            return

        g = self.grid
        T_np = g.T.to_numpy()
        fl_np = g.f_l.to_numpy()
        ux_np = g.ux.to_numpy()
        uz_np = g.uz.to_numpy()

        grid_pv = pv.ImageData()
        grid_pv.dimensions = (g.nx + 1, g.ny + 1, g.nz + 1)
        grid_pv.spacing = (g.dx * 1000,) * 3
        grid_pv.cell_data["Temperature_K"] = T_np.ravel(order="F")
        grid_pv.cell_data["Liquid_Fraction"] = fl_np.ravel(order="F")
        grid_pv.cell_data["T_max_K"] = g.T_max.to_numpy().ravel(order="F")
        grid_pv.cell_data["Velocity_X_ms"] = (ux_np * g.dx / g.dt).ravel(order="F")
        grid_pv.cell_data["Velocity_Z_ms"] = (uz_np * g.dx / g.dt).ravel(order="F")
        grid_pv.save(path)
        print(f"[WAAMTwin] VTK exported → {path}")

    def export_surface_vtk(self, path: str = "bead_surface.vtp") -> None:
        """Export φ=0.5 isosurface as PolyData (melt-pool boundary mesh)."""
        if __import__("os").environ.get("WAAM_HEADLESS") == "1":
            return
        try:
            import pyvista as pv
        except ImportError:
            print("[WAAMTwin] pyvista not installed. Skipping surface VTK.")
            return
        g = self.grid
        phi_np = g.phi.to_numpy()
        grid_pv = pv.ImageData()
        grid_pv.dimensions = (g.nx + 1, g.ny + 1, g.nz + 1)
        grid_pv.spacing = (g.dx * 1000,) * 3
        grid_pv.origin = (self._window_offset_x_m * 1000, 0.0, 0.0)
        grid_pv.cell_data["phi"] = phi_np.ravel(order="F")
        grid_pv.cell_data["Liquid_Fraction"] = g.f_l.to_numpy().ravel(order="F")
        grid_pv.cell_data["Temperature_K"] = g.T.to_numpy().ravel(order="F")
        point_grid = grid_pv.cell_data_to_point_data()
        surf = None
        for scalar in ("phi", "Liquid_Fraction"):
            try:
                candidate = point_grid.contour([0.5], scalars=scalar)
            except Exception:
                candidate = point_grid.contour(isosurfaces=[0.5], scalars=scalar)
            if candidate.n_cells > 0:
                surf = candidate
                break
        if surf is None or surf.n_cells == 0:
            print("[WAAMTwin] No φ/f_l=0.5 surface found; skipping surface VTK.")
            return
        surf.save(path)
        print(f"[WAAMTwin] Surface VTK exported → {path}  ({surf.n_cells} cells)")
