"""
twin.py — WAAMTwin v2 orchestrator
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np

from .grid import WAAMGrid
from .materials import MaterialProps, load_material
from .gpu_tables import MaterialGPUTables
from .physics.arc import create_heat_source
from .physics.bead_geometry import (
    bead_reinforcement_height_mm,
    bead_reinforcement_width_mm,
    estimate_toe_angle_deg,
)
from .physics.deposition import (
    droplet_mass_kg,
    expected_deposited_mass_kg,
    infer_transfer_mode,
    wire_mass_flux_kg_s,
)
from .physics.electrical_stickout import droplet_entry_temperature_K
from .physics.weld_forces import arc_pressure_peak_pa, droplet_impact_velocity_m_s
from . import kernels
from . import logging_util as log
from .solvers.coupled_step import coupled_step

# Lattice reference density. All LBM fields use ρ_lu ≈ 1; physical density
# enters only through the unit conversions (force_scale etc.). Initialising
# the lattice at the physical density (as previously done) silently rescaled
# every Guo body force by 1/ρ_phys ≈ 1.3e-4 — killing Marangoni convection.
RHO_LATTICE = 1.0


def _same_thermal_alloy(a: MaterialProps, b: MaterialProps) -> bool:
    """True when plate and wire share the thermal constants the dual path would split."""
    return (
        abs(a.T_solidus - b.T_solidus) < 0.5
        and abs(a.T_liquidus - b.T_liquidus) < 0.5
        and abs(a.k - b.k) < 1e-9
        and abs(a.cp - b.cp) < 1e-9
        and abs(a.rho - b.rho) < 1e-6
        and abs(a.L_fusion - b.L_fusion) < 1.0
    )


class WAAMTwin:
    """GPU-resident WAAM melt-pool digital twin (Taichi LBM + enthalpy-porosity)."""

    def __init__(
        self,
        material: str | MaterialProps = "ER70S-6",
        plate_material: str | MaterialProps | None = None,
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
        enable_marangoni: bool = True,
        enable_buoyancy: bool = True,
        enable_arc_pressure: bool = True,
        enable_deposition_momentum: bool = True,
        enable_lorentz: bool = False,
        enable_gas_shear: bool = False,
        enable_droplet_impact_pressure: bool = True,
        use_recoil_clausius_clapeyron: bool = True,
        enable_substrate_growth: bool = False,
        enable_moving_window: bool = False,
        enable_alloy_mixing: bool = False,
        enable_wetting: bool = False,
        enable_hydrostatic_gravity: bool = False,
        enable_bead_freeze: bool = False,
        enable_ctwd: bool = False,
        contact_angle_deg: float | None = None,
        deposition_superheat_K: float = 500.0,
        deposition_footprint_sigma_scale: float = 1.0,
        stickout_mm: float = 12.0,
        ctwd_mm: float = 15.0,
        layer_height_mm: float = 1.2,
        recoil_pressure_pa: float = 5000.0,
        welding_current_A: float = 140.0,
        sigma_liquid_Sm: float = 1.0e6,
        sigma_solid_Sm: float = 1.0e5,
        lorentz_jacobi_iters: int = 40,
        lorentz_jacobi_omega: float = 0.85,
        gas_jet_velocity_m_s: float = 12.0,
        gas_shear_coeff: float = 1.0,
        T_boiling_K: float = 3100.0,
        P_vapor_ref_Pa: float = 101325.0,
        L_vapor_J_kg: float = 6.0e6,
        R_spec_vapor_J_kgK: float = 450.0,
        travel_speed_m_s: float = 0.0,
        use_variable_tau: bool = True,
        enable_enthalpy_cap: bool = True,
        enable_evaporative_cooling: bool = False,
        T_vapor_cap_K: float = 3200.0,
        arc_surface_weighting: bool = True,
        arc_penetration_mm: float = 2.0,
        wire_diameter_mm: float = 1.2,
        arc_sigma_mm: float = 2.0,
        bulk_tau: float | None = None,
        dt_scale: float = 1.0,
        warn_on_force_clamp: bool = False,
    ):
        # ── Basic physical-plausibility validation ────────────────────────
        if arc_power_W < 0:
            raise ValueError(f"arc_power_W must be >= 0, got {arc_power_W}")
        if not (0.0 < arc_efficiency <= 1.0):
            raise ValueError(f"arc_efficiency must be in (0, 1], got {arc_efficiency}")
        if wire_diameter_mm <= 0:
            raise ValueError(f"wire_diameter_mm must be > 0, got {wire_diameter_mm}")
        if arc_sigma_mm <= 0:
            raise ValueError(f"arc_sigma_mm must be > 0, got {arc_sigma_mm}")
        if contact_angle_deg is not None and not (0.0 < contact_angle_deg < 180.0):
            raise ValueError(f"contact_angle_deg must be in (0, 180), got {contact_angle_deg}")

        if isinstance(material, MaterialProps):
            self.mat = material
        else:
            self.mat = load_material(material, high_sulphur)

        if max_tracers is None:
            max_tracers = 50_000

        self.Q_w = arc_power_W
        self.eta = arc_efficiency
        self.T_amb = T_ambient
        self.use_srt = use_srt
        self.C_darcy = C_darcy
        self.arc_pressure = arc_pressure_pa
        self.arc_pressure_model = "constant"  # or lin_eagar
        self.pressure_sigma_m: float | None = None  # None → use arc_sigma_m
        self.physics_tier = "flow"
        self.strict_mode = False
        self.enable_force_diagnostics = True
        self._force_diag: dict[str, float] = {}
        self._last_arc_ijk: tuple[float, float, float] | None = None
        self.recoil_accommodation = 0.54
        # LBM stability caps (Ma ≲ 0.15 with c_s = 1/√3 ≈ 0.577)
        self.u_mach_limit_lu = 0.08
        self.force_limit_lu = 0.05
        self._lorentz_unconverged = 0
        self._lorentz_unconverged_streak = 0
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
        self.enable_marangoni = enable_marangoni
        self.enable_buoyancy = enable_buoyancy
        self.enable_arc_pressure = enable_arc_pressure
        self.enable_deposition_momentum = enable_deposition_momentum
        self.enable_lorentz = enable_lorentz
        self.enable_gas_shear = enable_gas_shear
        # Grid after feature flags we care about for optional allocation.
        self.grid = WAAMGrid(
            nx, ny, nz, dx, self.mat,
            max_tracers=max_tracers,
            allocate_lorentz=enable_lorentz,
            allocate_vof=enable_vof,
            dt_scale=dt_scale,
        )
        self.warn_on_force_clamp = bool(warn_on_force_clamp)
        self.enable_droplet_impact_pressure = enable_droplet_impact_pressure
        self.use_recoil_clausius_clapeyron = use_recoil_clausius_clapeyron
        self.enable_substrate_growth = enable_substrate_growth
        self.enable_moving_window = enable_moving_window
        self.enable_alloy_mixing = enable_alloy_mixing
        self.alloy_mix_rate = 0.25
        self.enable_wetting = enable_wetting
        self.enable_hydrostatic_gravity = enable_hydrostatic_gravity
        self.enable_bead_freeze = enable_bead_freeze
        self.enable_ctwd = enable_ctwd
        self.contact_angle_deg = (
            contact_angle_deg if contact_angle_deg is not None else self.mat.contact_angle_deg
        )
        self.theta_rad = math.radians(self.contact_angle_deg)
        self.deposition_superheat_K = deposition_superheat_K
        self.deposition_footprint_sigma_scale = deposition_footprint_sigma_scale
        self.stickout_m = stickout_mm * 1e-3
        self.ctwd_nominal_m = ctwd_mm * 1e-3
        self.ctwd_m = ctwd_mm * 1e-3
        self.layer_height_m = layer_height_mm * 1e-3
        self.rho_e_ohm_m = self.mat.rho_e_ohm_m
        self.eta_stick = self.mat.eta_stick
        self.recoil_pressure_pa = recoil_pressure_pa
        self.welding_current_A = welding_current_A
        self.sigma_liquid_Sm = sigma_liquid_Sm
        self.sigma_solid_Sm = sigma_solid_Sm
        self.lorentz_jacobi_iters = lorentz_jacobi_iters
        self.lorentz_jacobi_omega = lorentz_jacobi_omega
        self.lorentz_jacobi_tol = 1.0e-4
        self.lorentz_jacobi_cold_iters: int | None = None
        self._lorentz_iters_last = 0
        self._lorentz_max_iters_last = 0
        self.gas_jet_velocity_m_s = gas_jet_velocity_m_s
        self.gas_shear_coeff = gas_shear_coeff
        self.T_boiling_K = T_boiling_K
        self.T_recoil_onset_K = None  # None → max(0.85·T_boil, T_liq+200)
        self.P_vapor_ref_Pa = P_vapor_ref_Pa
        self.L_vapor_J_kg = L_vapor_J_kg
        self.R_spec_vapor_J_kgK = R_spec_vapor_J_kgK
        self.travel_speed_m_s = travel_speed_m_s
        self.marangoni_scale = 1.0
        self.use_variable_tau = use_variable_tau
        self.enable_enthalpy_cap = enable_enthalpy_cap
        self.enable_evaporative_cooling = enable_evaporative_cooling
        self.evap_cooling_scale = 25.0
        self._evap_energy_J_step = 0.0
        self._evap_energy_J_cum = 0.0
        self.T_vapor_cap_K = T_vapor_cap_K
        self.arc_surface_weighting = arc_surface_weighting
        self.arc_penetration_m = arc_penetration_mm * 1e-3
        self.wire_diameter_m = wire_diameter_mm * 1e-3
        self.wire_feed_m_s = 0.0
        self.droplet_transfer_mode = "auto"
        self.pulse_frequency_hz = 0.0
        self.droplet_size_jitter = 0.12
        self.impact_lead_angle_deg = 0.0
        self.trailing_solidify_lookback_mm = 2.5
        self.trailing_solidify_temp_margin_K = 35.0
        self._deposited_volume_m3 = 0.0
        self._n_droplets_fired = 0
        self._deposition_overflow = 0
        self._welding_time_s = 0.0
        self._bead_height_m = 0.0
        # Substrate thickness: None → legacy nz//5. Job may set plate_thickness_mm.
        self.plate_thickness_mm: float | None = None
        # Lateral footprint: None → full domain XY. size=(Lx,Ly) mm, origin=(x0,y0) mm.
        self.plate_size_mm: tuple[float, float] | None = None
        self.plate_origin_mm: tuple[float, float] | None = None
        self.nz_solid = max(1, nz // 5)
        self._step_n = 0
        self._last_droplet_time = 0.0
        self._window_offset_x_m = 0.0
        self._window_offset_y_m = 0.0
        self._window_offset_z_m = 0.0
        self._interpass_cooling_steps = 0
        self._interpass_travel_m_s = None
        self.preset_name: str | None = None
        self.probe_recorder = None
        self._job_path: str | None = None
        self.use_torch_z = False
        self.substrate_z_m = 0.0
        self.frame_origin_mm = (0.0, 0.0, 0.0)
        self._sim_origin_offset_x_m = 0.0
        self._sim_origin_offset_y_m = 0.0
        self._sim_origin_offset_z_m = 0.0
        self.weld_frame = None
        self._last_torch_pos_m: tuple[float, float, float] | None = None
        self._torch_dir_xyz = (1.0, 0.0, 0.0)
        self._warned_zero_wire_feed = False
        self._warned_overflow = False
        self._force_clamp_hits_step = 0
        self._mach_clamp_hits_step = 0
        self._force_clamp_hits_cum = 0
        self._mach_clamp_hits_cum = 0
        self._force_clamp_steps = 0
        self._mach_clamp_steps = 0

        kernels.bind_velocity_set(self.grid)

        g = self.grid
        mat = self.mat

        self.cp_rho = mat.rho * mat.cp
        self.L_rho = mat.rho * mat.L_fusion
        self.alpha_lu = mat.alpha * g.dt / (g.dx ** 2)
        self.omega = 1.0 / g.tau
        self.omega_bulk = 1.0 / bulk_tau if bulk_tau else self.omega
        # Arc Gaussian sigma is a process parameter (job-configurable), not a
        # constant: previously hardcoded at 2 mm regardless of job settings.
        self.arc_sigma_m = arc_sigma_mm * 1e-3
        self.sigma_cells = self.arc_sigma_m / g.dx
        self.g_lu = 9.81 * (g.dt ** 2) / g.dx
        self.beta_T = mat.beta_T

        # Unit conversion: physical force density f [N/m³] → lattice
        # acceleration a_lu = (f/ρ)·dt²/dx. With ρ_lu = 1, a_lu is numerically
        # the lattice force density used by the Guo forcing terms.
        scale_F = (g.dt ** 2) / (mat.rho * g.dx ** 2)
        self.force_scale = scale_F
        # Marangoni CSF: F = dγ/dT · ∇_s T · |∇φ|. Both gradients are in
        # lattice units (per-cell), so the prefactor needs scale_F/dx — same
        # 1/dx as gamma_lu below. (The missing 1/dx underweighted Marangoni
        # by ~3–4 orders of magnitude.)
        self.dgamma_dT_lu = mat.dgamma_dT * scale_F / g.dx
        self.gamma_lu = mat.gamma_0 * scale_F / g.dx
        self.droplet_vz_lu = -0.03

        self.gpu_tables = MaterialGPUTables(mat)
        self.use_material_tables = self.gpu_tables.enabled
        cp_sol = mat.cp_at(mat.T_solidus)
        self.H_sol = mat.rho * cp_sol * mat.T_solidus
        self.H_liq = self.H_sol + self.L_rho
        self.arc_source = create_heat_source(heat_source)
        self._configure_dual_alloy(plate_material)

        log.info(f"[WAAMTwin] Material: {mat.name}  status={mat.status}")
        if self.use_material_tables:
            parts = ["cp", "k", "dγ/dT"]
            if self.use_variable_tau and self.mat.tables.mu:
                parts.append("μ→τ")
            log.info(f"[WAAMTwin] T-dependent tables: {', '.join(parts)} on GPU")
        if self.enable_vof:
            log.info("[WAAMTwin] VOF advection enabled")
        log.info(f"[WAAMTwin] Heat source: {heat_source}")
        log.info(f"[WAAMTwin] VRAM estimate: {g.estimated_vram_mb():.1f} MB")
        log.info(f"[WAAMTwin] α_lu={self.alpha_lu:.6f}  τ_T={g.tau_T:.4f}")
        log.info(f"[WAAMTwin] Collision: {'SRT' if use_srt else 'MRT (two-rate)'}")

    def _configure_dual_alloy(self, plate_material: str | MaterialProps | None) -> None:
        """Optional substrate alloy. Opt-in: same-alloy jobs keep the wire-only path."""
        self.plate_mat: MaterialProps | None = None
        self.use_dual_alloy = False
        self.gpu_tables_plate = None
        self.plate_cp_rho = self.cp_rho
        self.plate_L_rho = self.L_rho
        self.plate_alpha_lu = self.alpha_lu
        self.plate_H_sol = self.H_sol
        self.plate_H_liq = self.H_liq
        self.plate_tau = self.grid.tau
        self.plate_force_scale = self.force_scale
        self.plate_dgamma_dT_lu = self.dgamma_dT_lu
        if plate_material is None:
            return
        if isinstance(plate_material, MaterialProps):
            pmat = plate_material
        else:
            pmat = load_material(plate_material)
        if _same_thermal_alloy(self.mat, pmat):
            log.info(
                f"[WAAMTwin] plate.material {pmat.name} matches wire {self.mat.name} "
                f"— single-alloy path"
            )
            return
        g = self.grid
        self.plate_mat = pmat
        self.use_dual_alloy = True
        self.gpu_tables_plate = MaterialGPUTables(pmat)
        self.plate_cp_rho = pmat.rho * pmat.cp_at(self.T_amb)
        self.plate_L_rho = pmat.rho * pmat.L_fusion
        self.plate_alpha_lu = pmat.alpha * g.dt / (g.dx ** 2)
        cp_sol = pmat.cp_at(pmat.T_solidus)
        self.plate_H_sol = pmat.rho * cp_sol * pmat.T_solidus
        self.plate_H_liq = self.plate_H_sol + self.plate_L_rho
        nu_phys = pmat.mu / pmat.rho
        nu_lu = nu_phys * g.dt / (g.dx ** 2)
        self.plate_tau = 3.0 * nu_lu + 0.5
        if self.plate_tau <= 0.505:
            log.warning(
                f"[WAAMTwin] plate τ={self.plate_tau:.4f} is marginally stable; "
                f"melted substrate cells may need a coarser dx or higher μ"
            )
        self.plate_force_scale = (g.dt ** 2) / (pmat.rho * g.dx ** 2)
        self.plate_dgamma_dT_lu = pmat.dgamma_dT * self.plate_force_scale / g.dx
        log.info(
            f"[WAAMTwin] Dual alloy: wire={self.mat.name}  plate={pmat.name}  "
            f"T_sol {self.mat.T_solidus:.0f}/{pmat.T_solidus:.0f} K  "
            f"(enable_alloy_mixing={'on' if self.enable_alloy_mixing else 'off, birth composition'})"
        )

    def resolve_nz_solid(self, test_fluid_domain: bool = False) -> int:
        """Substrate layer count from job plate thickness, else nz//5.

        Always leaves bead/air headroom above the plate so ``feed_wire_surface``
        has gas cells to convert. A job thickness that exceeds the grid is
        clamped (with a warning) rather than filling ``nz-2``.
        """
        if test_fluid_domain:
            return 0
        g = self.grid
        # Need room for droplet footprint above arc_k (several cells of gas).
        headroom = max(6, g.nz // 4)
        max_solid = max(1, g.nz - headroom)
        thick = getattr(self, "plate_thickness_mm", None)
        if thick is not None and float(thick) > 0.0:
            n = int(round(float(thick) / (g.dx * 1000.0)))
            if n > max_solid:
                from . import logging_util as log
                log.warning(
                    f"[WAAMTwin] plate_thickness_mm={float(thick):.1f} needs {n} cells but "
                    f"grid only allows {max_solid} (nz={g.nz}, headroom={headroom}) — "
                    f"clamping so deposition has air above the plate."
                )
            return max(1, min(n, max_solid))
        return max(1, min(g.nz // 5, max_solid))

    def resolve_plate_ij(self) -> tuple[int, int, int, int]:
        """Inclusive/exclusive plate footprint indices (i0,i1,j0,j1).

        Default: full domain. With ``plate_size_mm``, coupon is centered unless
        ``plate_origin_mm`` sets the lower-left corner in domain millimetres.
        """
        g = self.grid
        dx_mm = g.dx * 1000.0
        if not getattr(self, "plate_size_mm", None):
            return 0, g.nx, 0, g.ny

        lx, ly = float(self.plate_size_mm[0]), float(self.plate_size_mm[1])
        ni = max(1, int(round(lx / dx_mm)))
        nj = max(1, int(round(ly / dx_mm)))
        ni = min(ni, g.nx)
        nj = min(nj, g.ny)

        if getattr(self, "plate_origin_mm", None) is not None:
            x0, y0 = float(self.plate_origin_mm[0]), float(self.plate_origin_mm[1])
            i0 = int(round(x0 / dx_mm))
            j0 = int(round(y0 / dx_mm))
        else:
            i0 = (g.nx - ni) // 2
            j0 = (g.ny - nj) // 2

        i0 = max(0, min(i0, g.nx - ni))
        j0 = max(0, min(j0, g.ny - nj))
        return i0, i0 + ni, j0, j0 + nj

    def apply_plate_geometry(
        self,
        *,
        plate_thickness_mm: float | None = None,
        plate_size_mm: tuple[float, float] | None = None,
        plate_origin_mm: tuple[float, float] | None = None,
    ) -> None:
        """Store job plate geometry used by ``reset()`` / ``init_grid``."""
        if plate_thickness_mm is not None:
            self.plate_thickness_mm = float(plate_thickness_mm)
            self.nz_solid = self.resolve_nz_solid()
        if plate_size_mm is not None:
            self.plate_size_mm = (float(plate_size_mm[0]), float(plate_size_mm[1]))
        if plate_origin_mm is not None:
            self.plate_origin_mm = (float(plate_origin_mm[0]), float(plate_origin_mm[1]))

    @classmethod
    def from_preset(
        cls,
        preset: str = "standard",
        material: str = "ER70S-6",
        plate_material: str | MaterialProps | None = None,
        domain_mm: tuple[float, float, float] | list[float] | None = None,
        dx_mm: float | None = None,
        plate_thickness_mm: float | None = None,
        **kwargs: Any,
    ) -> "WAAMTwin":
        from .runtime import (
            DEMO_DEFAULT_DOMAIN_MM,
            auto_grid,
            auto_tracer_count,
            check_vram_budget,
            ensure_taichi,
            resolve_grid_budget_mb,
            resolve_preset,
        )

        ensure_taichi()
        cfg = resolve_preset(preset)
        vram = resolve_grid_budget_mb(cfg)
        tracers = auto_tracer_count(vram, cfg)
        dom = DEMO_DEFAULT_DOMAIN_MM if domain_mm is None else (
            float(domain_mm[0]), float(domain_mm[1]), float(domain_mm[2])
        )
        dx_target = cfg.target_dx_mm if dx_mm is None else float(dx_mm)
        vram_lorentz = bool(kwargs.pop("vram_lorentz", True))
        vram_vof = bool(kwargs.pop("vram_vof", True))
        vram_export = bool(kwargs.pop("vram_export", True))
        kwargs.setdefault("dt_scale", 1.0)
        nx, ny, nz, dx = auto_grid(
            dom, dx_target, vram, tracers, max_cells=cfg.max_cells,
            lorentz=vram_lorentz, vof=vram_vof, export=vram_export,
        )
        check_vram_budget(
            nx, ny, nz, tracers, vram,
            lorentz=vram_lorentz, vof=vram_vof, export=vram_export,
        )

        twin = cls(
            material=material,
            plate_material=plate_material,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            use_srt=cfg.use_srt,
            max_tracers=tracers,
            **kwargs,
        )
        twin.preset_name = cfg.name
        if plate_thickness_mm is not None:
            twin.plate_thickness_mm = float(plate_thickness_mm)
            twin.nz_solid = twin.resolve_nz_solid()
        i0, i1, j0, j1 = twin.resolve_plate_ij()
        log.info(
            f"[WAAMTwin] hardware={cfg.name}  grid={nx}×{ny}×{nz}  dx={dx*1e3:.3f}mm"
            f"  domain={dom[0]:.0f}×{dom[1]:.0f}×{dom[2]:.0f}mm"
            f"  budget≈{vram} MB"
            + (f"  max_cells={cfg.max_cells}" if cfg.max_cells else "")
            + f"  plate≈{twin.nz_solid * dx * 1e3:.2f}mm thick"
            f"  footprint i=[{i0},{i1}) j=[{j0},{j1})"
        )
        return twin

    @classmethod
    def from_job(
        cls,
        job_path: str | pathlib.Path,
        preset_override: str | None = None,
        **kwargs: Any,
    ) -> "WAAMTwin":
        from .job import load_job_config, resolve_plate_and_domain, JobConfig
        from .runtime import ensure_taichi, resolve_preset

        ensure_taichi()
        job = load_job_config(job_path)
        job_cfg = JobConfig.from_dict(job)
        if preset_override:
            # Switch hardware profile only. Job domain_mm / plate / dx request
            # stay intact; auto_grid may coarsen dx to fit the profile budget.
            sim = job.setdefault("simulation", {})
            sim["preset"] = preset_override
            log.info(
                f"[WAAMTwin] hardware override → {preset_override} "
                f"(job domain/plate kept; dx may coarsen to fit budget)"
            )
        preset = job.get("simulation", {}).get("preset", "standard")
        material = job.get("material", "ER70S-6")
        plate_material = (job.get("plate") or {}).get("material")
        process = job.get("process", {})

        # Arc electrical power = V·I·PF·duty. ASSUMPTION: voltage_V and
        # current_A are average (RMS-equivalent) values of the same waveform.
        # For pulsed transfer supply duty_cycle; for AC processes supply
        # power_factor — otherwise V(rms)×I(peak) style mismatches inflate
        # power by 20–40% with no warning.
        from .job import electrical_arc_power_W
        arc_w = electrical_arc_power_W(process, required=True)
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

        cfg = resolve_preset(preset)
        from .runtime import DEMO_DEFAULT_DOMAIN_MM
        grid = resolve_plate_and_domain(
            job,
            cfg.target_dx_mm,
            fallback_domain_mm=DEMO_DEFAULT_DOMAIN_MM,
        )

        from .runtime import vram_flags_from_job
        v_lorentz, v_vof, v_export = vram_flags_from_job(job)
        sim_job = job.get("simulation") or {}
        dt_scale = float(kwargs.pop("dt_scale", sim_job.get("dt_scale", 1.0)))
        twin = cls.from_preset(
            preset=preset,
            material=material,
            plate_material=plate_material,
            domain_mm=grid["domain_mm"],
            dx_mm=grid["dx_mm"],
            plate_thickness_mm=grid["plate_thickness_mm"],
            arc_power_W=arc_w,
            arc_efficiency=arc_eta,
            T_ambient=T_amb,
            heat_source=str(job.get("heat_source", kwargs.pop("heat_source", "gaussian2d"))),
            vram_lorentz=v_lorentz,
            vram_vof=v_vof,
            vram_export=v_export,
            dt_scale=dt_scale,
            **heat_kwargs,
            **kwargs,
        )
        from .job import apply_job_to_twin
        apply_job_to_twin(twin, job_cfg)
        twin.apply_plate_geometry(
            plate_thickness_mm=grid["plate_thickness_mm"],
            plate_size_mm=grid["plate_size_mm"],
            plate_origin_mm=grid["plate_origin_mm"],
        )
        # Footprint log after plate geometry is applied (from_preset ran earlier).
        i0, i1, j0, j1 = twin.resolve_plate_ij()
        dx_mm = twin.grid.dx * 1e3
        ps = twin.plate_size_mm
        plate_xy = (
            f"{ps[0]:.0f}×{ps[1]:.0f} mm" if ps else "full domain XY"
        )
        log.info(
            f"[WAAMTwin] job geometry: plate {plate_xy}  "
            f"footprint i=[{i0},{i1}) j=[{j0},{j1})  "
            f"resolved dx={dx_mm:.3f} mm"
        )
        twin._job_config = job
        twin._job_path = str(job_path)
        probes_cfg = job.get("probes")
        if probes_cfg:
            from .export.probes import ProbeRecorder
            twin.probe_recorder = ProbeRecorder.from_job_list(probes_cfg, twin)
        return twin

    def reset(self, T_ambient: float | None = None, test_fluid_domain: bool = False):
        T0 = T_ambient or self.T_amb
        g = self.grid
        if test_fluid_domain:
            nz_solid = -1
            self.nz_solid = 0
            i0, i1, j0, j1 = 0, g.nx, 0, g.ny
        else:
            self.nz_solid = self.resolve_nz_solid(False)
            nz_solid = self.nz_solid
            i0, i1, j0, j1 = self.resolve_plate_ij()

        kernels.init_grid(
            g.f_a, g.rho, g.ux, g.uy, g.uz,
            g.T, g.H, g.f_l, g.phi, g.flags, g.alloy_id, g.alloy_frac,
            T0, RHO_LATTICE, self.cp_rho, float(getattr(self, "plate_cp_rho", self.cp_rho)),
            nz_solid,
            i0, i1, j0, j1,
            g.FLAG_FLUID, g.FLAG_SOLID, g.FLAG_GAS,
        )
        kernels.init_aux_fields(
            g.T_max, g.T_prev, g.dT_dt,
            g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
            g.Fx_snap, g.Fy_snap, g.Fz_snap,
            g.Fx, g.Fy, g.Fz, T0,
            g.porosity_active, g.tracer_head, g.max_tracers,
        )
        kernels.stream(
            g.f_a, g.f_b, g.flags,
            g.FLAG_SOLID, g.FLAG_GAS,
            g.nx, g.ny, g.nz,
        )
        self._step_n = 0
        self._last_droplet_time = 0.0
        self._deposited_volume_m3 = 0.0
        self._n_droplets_fired = 0
        self._deposition_overflow = 0
        self._welding_time_s = 0.0
        self._bead_height_m = 0.0
        self._last_torch_pos_m = None
        self._torch_dir_xyz = (1.0, 0.0, 0.0)
        self._warned_zero_wire_feed = False
        self._warned_overflow = False
        self._lorentz_warm = False
        self._lorentz_unconverged = 0
        self._lorentz_unconverged_streak = 0
        self._lorentz_iters_last = 0
        self._lorentz_max_iters_last = 0
        self._window_offset_x_m = 0.0
        self._window_offset_y_m = 0.0
        self._window_offset_z_m = 0.0
        self._evap_energy_J_step = 0.0
        self._evap_energy_J_cum = 0.0
        self._force_clamp_hits_step = 0
        self._mach_clamp_hits_step = 0
        self._force_clamp_hits_cum = 0
        self._mach_clamp_hits_cum = 0
        self._force_clamp_steps = 0
        self._mach_clamp_steps = 0
        g.deposit_vol_buf[None] = 0.0
        g.deposit_real_buf[None] = 0.0
        log.info(f"[WAAMTwin] Grid reset. T_ambient={T0:.1f}K  test_fluid={test_fluid_domain}")

    def step(
        self,
        torch_x_m: float,
        torch_y_m: float,
        is_welding: bool = True,
        torch_z_m: float | None = None,
    ):
        z_now = 0.0 if torch_z_m is None else float(torch_z_m)
        if self._last_torch_pos_m is not None:
            px, py, pz = self._last_torch_pos_m
            dx = torch_x_m - px
            dy = torch_y_m - py
            dz = z_now - pz
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            if norm > 1e-12:
                self._torch_dir_xyz = (dx / norm, dy / norm, dz / norm)
        self._last_torch_pos_m = (torch_x_m, torch_y_m, z_now)
        if (
            is_welding
            and self.droplet_freq > 0
            and self.wire_feed_m_s <= 0.0
            and not self._warned_zero_wire_feed
        ):
            self._warned_zero_wire_feed = True
            log.warning(
                "[WAAMTwin] WARNING: welding step with wire_feed_m_s == 0 — "
                "no metal will be deposited. Set process.wire_feed_m_min in the "
                "job file or twin.wire_feed_m_s directly."
            )
        if self.enable_moving_window:
            self._maybe_shift_window(torch_x_m, torch_y_m, torch_z_m)
        ox, oy, oz = self.window_offset_m()
        sim_x = torch_x_m - ox - self._sim_origin_offset_x_m
        sim_y = torch_y_m - oy - self._sim_origin_offset_y_m
        # Wire-feed ledger: only arc-on time counts toward expected wire mass.
        if is_welding:
            self._welding_time_s = float(getattr(self, "_welding_time_s", 0.0)) + self.grid.dt
        z_sim = torch_z_m
        if z_sim is not None:
            z_sim = (
                float(z_sim)
                - float(getattr(self, "_sim_origin_offset_z_m", 0.0) or 0.0)
                - oz
            )
        coupled_step(self, sim_x, sim_y, is_welding, z_sim)
        self._check_strict_mode()

    def _check_strict_mode(self) -> None:
        """Abort on silent physics drift when strict_mode / WAAM_STRICT is set."""
        if not self.strict_mode:
            return
        streak = int(getattr(self, "_lorentz_unconverged_streak", 0) or 0)
        # Stricter than the historical 50-step allowance: a warm-started Jacobi
        # that fails for >10 consecutive steps is already polluting J×B.
        if self.enable_lorentz and streak > 10:
            raise RuntimeError(
                f"[strict_mode] Lorentz Jacobi failed to converge for {streak} "
                f"consecutive steps — raise lorentz_jacobi_iters or disable strict_mode"
            )
        # Sustained Mach/force clamping means the pool is numerically limited.
        if self._step_n >= 100:
            f_steps = int(getattr(self, "_force_clamp_steps", 0) or 0)
            u_steps = int(getattr(self, "_mach_clamp_steps", 0) or 0)
            if f_steps > 0.5 * self._step_n:
                raise RuntimeError(
                    f"[strict_mode] body-force clamp active on {f_steps}/{self._step_n} "
                    f"steps (force_limit_lu={self.force_limit_lu}) — refine grid or "
                    "raise force_limit_lu / lower force magnitudes"
                )
            if u_steps > 0.5 * self._step_n:
                raise RuntimeError(
                    f"[strict_mode] Mach velocity clamp active on {u_steps}/{self._step_n} "
                    f"steps (u_mach_limit_lu={self.u_mach_limit_lu}) — refine grid or "
                    "reduce driving forces"
                )
        diag = getattr(self, "_force_diag", None) or {}
        if diag:
            from .physics.force_diagnostics import diagnostics_have_nan
            if diagnostics_have_nan(diag):
                raise RuntimeError(
                    f"[strict_mode] NaN/Inf in force_diagnostics: {diag}"
                )
        if self._n_droplets_fired < 20:
            return
        dep_mass = self._deposited_volume_m3 * self.mat.rho
        exp_wire = expected_deposited_mass_kg(self)
        ratio = dep_mass / max(exp_wire, 1e-12)
        if not (0.95 <= ratio <= 1.05):
            raise RuntimeError(
                f"[strict_mode] mass_balance_ratio={ratio:.3f} outside [0.95, 1.05] "
                f"(deposited vs ṁ·t_weld) after {self._n_droplets_fired} droplets "
                f"(overflow_count={self._deposition_overflow})"
            )

    def window_offset_m(self) -> tuple[float, float, float]:
        return (
            float(self._window_offset_x_m),
            float(self._window_offset_y_m),
            float(self._window_offset_z_m),
        )

    def _window_field_args(self):
        g = self.grid
        return (
            g.f_a, g.f_b,
            g.T, g.H, g.f_l, g.phi, g.flags,
            g.T_max, g.T_prev, g.dT_dt,
            g.time_above_800_s, g.time_above_1100_s, g.time_above_solidus_s,
            g.rho, g.ux, g.uy, g.uz, g.Fx, g.Fy, g.Fz,
            g.cp_rho_field, g.alpha_lu_field, g.dgamma_lu_field, g.tau_field,
            g.alloy_id, g.alloy_frac,
            self.T_amb, self.cp_rho, float(getattr(self, "plate_cp_rho", self.cp_rho)),
            RHO_LATTICE, self.nz_solid,
            g.FLAG_SOLID, g.FLAG_GAS, g.FLAG_FLUID,
            g.nx, g.ny, g.nz,
        )

    def _maybe_shift_window(
        self,
        torch_x_world: float,
        torch_y_world: float = 0.0,
        torch_z_world: float | None = None,
    ) -> None:
        g = self.grid
        from . import kernels

        sim_x = torch_x_world - self._window_offset_x_m
        trigger_x = 0.62 * (g.nx - 8) * g.dx
        if sim_x >= trigger_x:
            n_shift = max(8, g.nx // 10)
            kernels.shift_simulation_window_x(n_shift, *self._window_field_args())
            self._window_offset_x_m += n_shift * g.dx
            kernels.shift_tracers(
                g.porosity_pos, g.porosity_active,
                n_shift * g.dx, 0.0, 0.0, g.max_tracers,
            )
            log.info(
                f"[WAAMTwin] Moving window +X −{n_shift} cells  "
                f"offset_x={self._window_offset_x_m*1e3:.2f}mm"
            )

        sim_y = torch_y_world - self._window_offset_y_m
        n_y = max(8, g.ny // 10)
        margin_y = n_y * g.dx
        if sim_y >= (g.ny * g.dx - margin_y) and n_y < g.ny:
            kernels.shift_simulation_window_y(n_y, 1, *self._window_field_args())
            self._window_offset_y_m += n_y * g.dx
            kernels.shift_tracers(
                g.porosity_pos, g.porosity_active,
                0.0, n_y * g.dx, 0.0, g.max_tracers,
            )
            log.info(
                f"[WAAMTwin] Moving window +Y −{n_y} cells  "
                f"offset_y={self._window_offset_y_m*1e3:.2f}mm"
            )
        elif sim_y <= margin_y and n_y < g.ny:
            kernels.shift_simulation_window_y(n_y, -1, *self._window_field_args())
            self._window_offset_y_m -= n_y * g.dx
            kernels.shift_tracers(
                g.porosity_pos, g.porosity_active,
                0.0, -n_y * g.dx, 0.0, g.max_tracers,
            )
            log.info(
                f"[WAAMTwin] Moving window −Y +{n_y} cells  "
                f"offset_y={self._window_offset_y_m*1e3:.2f}mm"
            )

        if torch_z_world is None or not getattr(self, "use_torch_z", False):
            return
        sim_z = float(torch_z_world) - self._window_offset_z_m
        n_z = max(4, g.nz // 10)
        margin_z = n_z * g.dx
        if sim_z >= (g.nz * g.dx - margin_z) and n_z < g.nz and self.nz_solid - n_z >= 1:
            kernels.shift_simulation_window_z(n_z, 1, *self._window_field_args())
            self._window_offset_z_m += n_z * g.dx
            self.nz_solid = max(1, self.nz_solid - n_z)
            kernels.shift_tracers(
                g.porosity_pos, g.porosity_active,
                0.0, 0.0, n_z * g.dx, g.max_tracers,
            )
            log.info(
                f"[WAAMTwin] Moving window +Z −{n_z} cells  "
                f"offset_z={self._window_offset_z_m*1e3:.2f}mm  nz_solid={self.nz_solid}"
            )
        elif sim_z <= margin_z and n_z < g.nz and self.nz_solid + n_z <= g.nz - 2:
            kernels.shift_simulation_window_z(n_z, -1, *self._window_field_args())
            self._window_offset_z_m -= n_z * g.dx
            self.nz_solid = min(g.nz - 2, self.nz_solid + n_z)
            kernels.shift_tracers(
                g.porosity_pos, g.porosity_active,
                0.0, 0.0, -n_z * g.dx, g.max_tracers,
            )
            log.info(
                f"[WAAMTwin] Moving window −Z +{n_z} cells  "
                f"offset_z={self._window_offset_z_m*1e3:.2f}mm  nz_solid={self.nz_solid}"
            )

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
            raise ValueError(
                "Job has no torch_path / torch_path_csv waypoints. "
                "Add a path or pass an explicit waypoint list."
            )
        driver = TorchPathDriver(waypoints, self.travel_speed_m_s)

        if n_steps is None:
            n_steps = max(400, int(driver.total_length / max(self.travel_speed_m_s * g.dt, 1e-12)))

        # Explicit argument wins over the job-file setting; previously the job
        # value silently overrode a non-zero interpass_steps argument.
        if interpass_steps:
            interpass = int(interpass_steps)
        else:
            interpass = int(getattr(self, "_interpass_cooling_steps", 0) or 0)
        prev_seg = -1

        # Seed the torch direction from the first path segment so the very
        # first step's Goldak front/rear asymmetry points along the actual
        # travel direction instead of the default +x.
        if driver.segments:
            s0 = driver.segments[0]
            dxs, dys, dzs = s0.x1 - s0.x0, s0.y1 - s0.y0, s0.z1 - s0.z0
            norm = math.sqrt(dxs * dxs + dys * dys + dzs * dzs)
            if norm > 1e-12:
                self._torch_dir_xyz = (dxs / norm, dys / norm, dzs / norm)

        def _clamp(x_m: float, y_m: float) -> tuple[float, float]:
            # Clamp in WORLD coordinates: with a moving window the valid range
            # is [offset, offset + n·dx], not [0, n·dx].
            ox, oy, _ = self.window_offset_m()
            cx, cy = clamp_torch_to_domain(x_m - ox, y_m - oy, g.nx, g.ny, g.dx)
            return cx + ox, cy + oy

        for step, x, y, z in driver.positions_for_steps(n_steps, g.dt):
            seg = driver.segment_index_at_distance(driver.distance_at_step(step, g.dt))
            if interpass > 0 and seg > prev_seg and prev_seg >= 0:
                park = driver.segment_end(prev_seg)
                px, py, pz = park if park else (x, y, z)
                self._idle_interpass((px, py, pz), (x, y, z), interpass, _clamp)
            prev_seg = seg
            cx, cy = _clamp(x, y)
            self.step(cx, cy, is_welding, torch_z_m=z)

    def _idle_interpass(
        self,
        start_xyz: tuple[float, float, float],
        end_xyz: tuple[float, float, float],
        cooling_steps: int,
        clamp_xy,
    ) -> int:
        """Idle between path segments: optional travel then cooling dwell.

        Returns the number of idle steps taken. When
        ``_interpass_travel_m_s`` is set, the torch interpolates from
        *start* to *end* then parks for ``cooling_steps``. Otherwise this
        is a park-at-start dwell (legacy).
        """
        x0, y0, z0 = start_xyz
        x1, y1, z1 = end_xyz
        v = getattr(self, "_interpass_travel_m_s", None)
        motion = 0
        if v is not None and float(v) > 0.0:
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            motion = max(0, int(dist / max(float(v) * self.grid.dt, 1e-18)))
        n_idle = 0
        for i in range(motion):
            t = (i + 1) / max(motion, 1)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            z = z0 + t * (z1 - z0)
            cx, cy = clamp_xy(x, y)
            self.step(cx, cy, is_welding=False, torch_z_m=z)
            n_idle += 1
        park_x, park_y, park_z = (x1, y1, z1) if motion > 0 else (x0, y0, z0)
        for _ in range(max(0, int(cooling_steps))):
            cx, cy = clamp_xy(park_x, park_y)
            self.step(cx, cy, is_welding=False, torch_z_m=park_z)
            n_idle += 1
        return n_idle

    @staticmethod
    def _vtk_imagedata_path(path: str) -> str:
        """PyVista ImageData requires .vti (legacy .vts is rewritten)."""
        p = pathlib.Path(path)
        if p.suffix.lower() in (".vts", ""):
            return str(p.with_suffix(".vti"))
        return path

    def export_haz_vtk(self, path: str = "haz_map.vti") -> None:
        """Export peak temperature (HAZ) field to VTK."""
        from .export.vtk_io import export_volume, TIER_CORE
        export_volume(self, path, tiers=(TIER_CORE,), crop_liquid=False)

    def get_telemetry(self) -> dict:
        """Pool/process telemetry.

        Pool width is the TRANSVERSE (y) extent of the melt pool measured at
        the x-slice through the pool centroid; depth is measured from the
        substrate top surface down. Aggregates are reduced on GPU; only a
        single y–z f_l slice is copied to host for W/D (previously ~6 full
        volume transfers ≈ 40 MB on a production grid).
        """
        g = self.grid
        kernels.telemetry_pool_reduce(
            g.T, g.f_l, g.phi, g.dT_dt, g.ux, g.uy, g.uz,
            g.telem_n, g.telem_T_peak, g.telem_cool, g.telem_u2,
            g.telem_i_sum, g.telem_i_min, g.telem_i_max, g.telem_T_global,
        )
        n_liquid = int(g.telem_n[None])
        kernels.count_near_vapor_cap(
            g.T, g.flags, g.telem_n_cap,
            float(self.T_vapor_cap_K), 1.0, g.FLAG_GAS,
        )
        n_at_vapor_cap = int(g.telem_n_cap[None])

        if n_liquid > 0:
            T_peak = float(g.telem_T_peak[None])
            peak_cool = float(g.telem_cool[None])
            i_c = int(round(float(g.telem_i_sum[None]) / n_liquid))
            i_c = max(0, min(g.nx - 1, i_c))
            kernels.extract_fl_yz_slice(g.f_l, g.telem_fl_slice, i_c, g.ny, g.nz)
            sect = g.telem_fl_slice.to_numpy() > 0.5
            if sect.any():
                y_idx = np.where(sect.any(axis=1))[0]
                pool_width_m = (y_idx[-1] - y_idx[0] + 1) * g.dx
                z_idx = np.where(sect.any(axis=0))[0]
                pool_depth_m = max(0, self.nz_solid - z_idx[0]) * g.dx
            else:
                pool_width_m = pool_depth_m = 0.0
            pool_length_m = (
                float(g.telem_i_max[None]) - float(g.telem_i_min[None]) + 1.0
            ) * g.dx
            u_max_phys = float(np.sqrt(float(g.telem_u2[None]))) * g.dx / g.dt
        else:
            T_peak = float(g.telem_T_global[None])
            peak_cool = float(g.telem_cool[None])
            pool_width_m = pool_depth_m = pool_length_m = u_max_phys = 0.0

        active = g.porosity_active.to_numpy()
        n_trapped = int((active == 2).sum())
        n_active = int((active == 1).sum())
        porosity_pct = 100.0 * n_trapped / max(n_trapped + n_active, 1)

        dep_mass = self._deposited_volume_m3 * self.mat.rho
        exp_mass = expected_deposited_mass_kg(self)
        m_drop = droplet_mass_kg(self)
        exp_drop_mass = self._n_droplets_fired * m_drop
        mass_ratio = dep_mass / max(exp_mass, 1e-12)

        bead_h = bead_reinforcement_height_mm(self, g)
        bead_w = bead_reinforcement_width_mm(self, g)
        toe_deg = estimate_toe_angle_deg(self, g) if self.enable_wetting else 0.0

        force_diag = dict(getattr(self, "_force_diag", {}) or {})
        if getattr(self, "enable_force_diagnostics", True) and self._step_n > 0:
            try:
                from .physics.force_diagnostics import sample_force_diagnostics
                force_diag = sample_force_diagnostics(self)
                self._force_diag = force_diag
            except Exception as exc:
                if self.strict_mode:
                    raise
                force_diag = {"error": str(exc)}

        return {
            "step": self._step_n,
            "sim_time_ms": round(self._step_n * g.dt * 1000, 4),
            "peak_temp_K": round(T_peak, 1),
            "peak_temp_C": round(T_peak - 273.15, 1),
            "T_vapor_cap_K": self.T_vapor_cap_K,
            "n_cells_at_vapor_cap": n_at_vapor_cap,
            "vapor_cap_saturated": bool(
                self.enable_enthalpy_cap and T_peak >= float(self.T_vapor_cap_K) - 1.0
            ),
            "enable_evaporative_cooling": bool(
                getattr(self, "enable_evaporative_cooling", False)
            ),
            "evap_energy_J_step": round(float(getattr(self, "_evap_energy_J_step", 0.0)), 6),
            "evap_energy_J_cum": round(float(getattr(self, "_evap_energy_J_cum", 0.0)), 4),
            "peak_cooling_rate_Ks": round(min(float(peak_cool), 1.0e5), 1),
            "peak_cooling_rate_raw_Ks": round(float(peak_cool), 1),
            "pool_width_mm": round(pool_width_m * 1000, 3),
            "pool_depth_mm": round(pool_depth_m * 1000, 3),
            "pool_length_mm": round(pool_length_m * 1000, 3),
            "n_liquid_cells": n_liquid,
            "marangoni_vel_ms": round(u_max_phys, 4),
            "material_status": self.mat.status,
            "material_name": self.mat.name,
            "plate_material_name": (
                None if not getattr(self, "plate_mat", None) else self.plate_mat.name
            ),
            "dual_alloy": bool(getattr(self, "use_dual_alloy", False)),
            "heat_source": self.heat_source_name,
            "material_tables": self.use_material_tables,
            "vof_enabled": self.enable_vof,
            "travel_speed_mm_s": round(self.travel_speed_m_s * 1000, 3),
            "variable_tau": self.use_variable_tau and self.use_material_tables,
            "porosity_pct": round(porosity_pct, 3),
            "n_trapped_tracers": n_trapped,
            "window_offset_x_mm": round(self._window_offset_x_m * 1000, 3),
            "window_offset_y_mm": round(self._window_offset_y_m * 1000, 3),
            "window_offset_z_mm": round(self._window_offset_z_m * 1000, 3),
            "deposited_mass_g": round(dep_mass * 1000, 4),
            "expected_wire_mass_g": round(exp_mass * 1000, 4),
            "expected_drop_mass_g": round(exp_drop_mass * 1000, 4),
            "n_droplets_fired": self._n_droplets_fired,
            "welding_time_s": round(float(getattr(self, "_welding_time_s", 0.0)), 6),
            "droplet_mass_mg": round(m_drop * 1e6, 3),
            "droplet_transfer_mode": infer_transfer_mode(self),
            "droplet_impact_velocity_ms": round(droplet_impact_velocity_m_s(self), 4),
            "mass_balance_ratio": round(mass_ratio, 3),
            "evap_cooling_scale": float(getattr(self, "evap_cooling_scale", 25.0)),
            "recoil_accommodation": float(getattr(self, "recoil_accommodation", 0.54)),
            "wire_mass_flux_g_s": round(wire_mass_flux_kg_s(self) * 1000, 4),
            "arc_surface_weighting": self.arc_surface_weighting,
            "enable_enthalpy_cap": self.enable_enthalpy_cap,
            "enable_wetting": self.enable_wetting,
            "enable_hydrostatic_gravity": self.enable_hydrostatic_gravity,
            "enable_bead_freeze": self.enable_bead_freeze,
            "enable_ctwd": self.enable_ctwd,
            "contact_angle_deg": round(self.contact_angle_deg, 1),
            "bead_height_mm": round(bead_h, 3),
            "bead_width_mm": round(bead_w, 3),
            "toe_angle_deg": round(toe_deg, 1),
            "ctwd_mm": round(self.ctwd_m * 1000, 3),
            "T_drop_K": round(droplet_entry_temperature_K(self), 1),
            "deposition_overflow_count": self._deposition_overflow,
            "physics_tier": getattr(self, "physics_tier", "flow"),
            "strict_mode": bool(getattr(self, "strict_mode", False)),
            "arc_pressure_model": getattr(self, "arc_pressure_model", "constant"),
            "arc_pressure_peak_pa": round(arc_pressure_peak_pa(self), 1),
            "enable_marangoni": bool(getattr(self, "enable_marangoni", True)),
            "enable_buoyancy": bool(getattr(self, "enable_buoyancy", True)),
            "enable_arc_pressure": bool(getattr(self, "enable_arc_pressure", True)),
            "enable_lorentz": self.enable_lorentz,
            "enable_gas_shear": self.enable_gas_shear,
            "lorentz_unconverged_count": int(getattr(self, "_lorentz_unconverged", 0) or 0),
            "lorentz_unconverged_streak": int(getattr(self, "_lorentz_unconverged_streak", 0) or 0),
            "force_clamp_hits_step": int(getattr(self, "_force_clamp_hits_step", 0) or 0),
            "mach_clamp_hits_step": int(getattr(self, "_mach_clamp_hits_step", 0) or 0),
            "force_clamp_hits_cum": int(getattr(self, "_force_clamp_hits_cum", 0) or 0),
            "mach_clamp_hits_cum": int(getattr(self, "_mach_clamp_hits_cum", 0) or 0),
            "force_clamp_steps": int(getattr(self, "_force_clamp_steps", 0) or 0),
            "mach_clamp_steps": int(getattr(self, "_mach_clamp_steps", 0) or 0),
            "u_mach_limit_lu": float(getattr(self, "u_mach_limit_lu", 0.08)),
            "force_limit_lu": float(getattr(self, "force_limit_lu", 0.05)),
            "force_diagnostics": force_diag,
        }

    def export_vtk(self, path: str = "weld_pool.vti"):
        """Convenience wrapper: core + derived tiers (see export_vtk_full)."""
        return self.export_vtk_full(path, tiers=(0, 3), crop_liquid=False)

    def export_vtk_full(
        self,
        path: str = "weld_pool_full.vti",
        tiers: tuple[int, ...] = (0, 1, 2, 3),
        crop_liquid: bool = False,
        *,
        allow_skip: bool | None = None,
    ) -> str | None:
        from .export import vtk_io
        tier_tuple = tuple(vtk_io.TIER_MAP.get(t, t) for t in tiers)
        return vtk_io.export_volume(
            self, path, tiers=tier_tuple, crop_liquid=crop_liquid,
            allow_skip=allow_skip,
        )

    def export_surface_vtk(
        self, path: str = "bead_surface.vtp", *, allow_skip: bool | None = None,
    ) -> None:
        """Export φ=0.5 isosurface as PolyData (melt-pool boundary mesh)."""
        from .export.vtk_io import export_surface
        export_surface(self, path, include_kappa=True, allow_skip=allow_skip)

    def export_tracers_vtk(
        self, path: str = "tracers.vtp", *, allow_skip: bool | None = None,
    ) -> None:
        from .export.vtk_io import export_tracers
        export_tracers(self, path, allow_skip=allow_skip)

    def export_research_bundle(
        self,
        out_dir: str | pathlib.Path,
        tag: str | None = None,
        tiers: tuple[int, ...] = (0, 1, 3),
        include_surface: bool = True,
        include_tracers: bool = True,
        crop_liquid: bool = False,
        *,
        allow_skip: bool | None = None,
    ) -> dict[str, str]:
        from .export.bundle import export_research_bundle
        return export_research_bundle(
            self,
            out_dir,
            tag=tag,
            tiers=tiers,
            include_surface=include_surface,
            include_tracers=include_tracers,
            job_path=self._job_path,
            crop_liquid=crop_liquid,
            allow_skip=allow_skip,
        )

    def export_research_sequence(self, *args, **kwargs):
        from .export.bundle import export_research_sequence
        return export_research_sequence(self, *args, job_path=self._job_path, **kwargs)
