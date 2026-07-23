"""Build WAAMTwin + torch motion for the interactive viewer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..job import load_job_config, parse_torch_path
from ..paths import PROJECT_ROOT, resolve_project_path
from ..torch_path import TorchPathDriver, clamp_torch_to_domain


@dataclass
class ViewerSession:
    """Twin instance, job metadata, and torch motion state."""

    twin: Any
    job_label: str
    path_driver: TorchPathDriver | None = None
    torch_x_m: float = 0.0
    torch_y_m: float = 0.0
    torch_z_m: float = 0.0
    torch_spd_m_s: float = 0.005
    x_min_m: float = 0.0
    x_max_m: float = 1.0
    _bounce_dir: int = 1
    _warned_path_end: bool = False
    _auto_exported_path_end: bool = False

    @property
    def uses_path(self) -> bool:
        return self.path_driver is not None

    def path_complete(self) -> bool:
        """True when a CSV/inline path has reached its final waypoint."""
        if not self.path_driver:
            return False
        sim_t = self.twin._step_n * self.twin.grid.dt
        return self.path_driver.is_complete(sim_t)

    def welding_active(self, want_weld: bool = True) -> bool:
        """Arc on only while traveling the path (off after path end)."""
        if not want_weld:
            return False
        if self.path_complete():
            return False
        return True

    def export_g_bundle(self, out_dir) -> str | None:
        """Same research bundle as viewer key ``G`` (tiers 0–3)."""
        import pathlib

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        twin = self.twin
        tag = f"step_{twin._step_n:06d}"
        bundle_dir = out_dir / f"bundle_{tag}"
        twin.export_research_bundle(
            bundle_dir,
            tag=tag,
            tiers=(0, 1, 2, 3),
            include_surface=True,
            include_tracers=True,
        )
        return str(bundle_dir)

    def maybe_auto_export_on_path_end(self, out_dir) -> str | None:
        """Once per run: when path completes, write a G-equivalent bundle."""
        if self._auto_exported_path_end or not self.path_complete():
            return None
        self._auto_exported_path_end = True
        try:
            path = self.export_g_bundle(out_dir)
            print(f"[viewer] Path end — auto export (same as G) → {path}")
            return path
        except Exception as exc:
            print(f"[viewer] Path-end auto export failed: {exc}")
            return None

    def reset_motion(self) -> None:
        """Reset simulation and torch position to job start."""
        self.twin.reset()
        self._warned_path_end = False
        self._auto_exported_path_end = False
        g = self.twin.grid
        margin = 4 * g.dx
        cy = (g.ny // 2) * g.dx
        self.torch_x_m = margin
        self.torch_y_m = cy
        self.torch_z_m = 0.0
        self.x_min_m = margin
        self.x_max_m = (g.nx - 4) * g.dx
        self._bounce_dir = 1
        if self.path_driver and self.path_driver.segments:
            x0, y0, z0 = self.path_driver.segments[0].x0, self.path_driver.segments[0].y0, self.path_driver.segments[0].z0
            self.torch_x_m, self.torch_y_m, self.torch_z_m = x0, y0, z0

    def torch_position_now(self) -> tuple[float, float, float]:
        if self.path_driver:
            sim_t = self.twin._step_n * self.twin.grid.dt
            return self.path_driver.position_at_time(sim_t)
        return self.torch_x_m, self.torch_y_m, self.torch_z_m

    def advance_physics(self, n_steps: int, is_welding: bool = True) -> None:
        g = self.twin.grid
        for _ in range(n_steps):
            welding = self.welding_active(is_welding)
            if (
                self.path_driver
                and not welding
                and is_welding
                and not self._warned_path_end
            ):
                self._warned_path_end = True
                print(
                    "[viewer] Torch path complete — arc OFF (cooling at end waypoint). "
                    "Press R to reset."
                )
            if self.path_driver:
                sim_t = self.twin._step_n * g.dt
                x, y, z = self.path_driver.position_at_time(sim_t)
                # Clamp in window-local coordinates (world x is valid up to
                # offset + nx·dx when the moving window has shifted).
                off = self.twin._window_offset_x_m
                cx, cy = clamp_torch_to_domain(x - off, y, g.nx, g.ny, g.dx)
                x, y = cx + off, cy
                self.torch_x_m, self.torch_y_m, self.torch_z_m = x, y, z
            else:
                self.torch_x_m += self._bounce_dir * self.torch_spd_m_s * g.dt
                if self.torch_x_m >= self.x_max_m or self.torch_x_m <= self.x_min_m:
                    self._bounce_dir *= -1
                    self.torch_x_m = max(self.x_min_m, min(self.torch_x_m, self.x_max_m))
            tz = self.torch_z_m if getattr(self.twin, "use_torch_z", False) else None
            self.twin.step(self.torch_x_m, self.torch_y_m, is_welding=welding, torch_z_m=tz)

    def offset_x_mm(self) -> float:
        return self.twin._window_offset_x_m * 1000.0

    def torch_mm(self) -> tuple[float, float, float]:
        """Torch path / TCP position in WORLD mm.

        Path waypoints are already world coordinates, so no offset is added
        (adding it double-counted the moving-window shift). Only the demo
        bounce mode tracks the torch in window-local coordinates.

        Note: CSV paths often store ``z_mm=0`` (substrate datum). That is
        *not* the contact tip height — use :meth:`torch_marker_mm` for the
        yellow tip glyph and :meth:`torch_surface_mm` for the arc attachment.
        """
        x, y, z = self.torch_position_now()
        if self.path_driver is None:
            x += self.twin._window_offset_x_m
        return x * 1000.0, y * 1000.0, z * 1000.0

    def torch_surface_mm(self) -> tuple[float, float, float]:
        """World mm at the free-surface metal under the torch (physics ``arc_k``)."""
        from ..physics import free_surface

        twin = self.twin
        g = twin.grid
        x_mm, y_mm, _ = self.torch_mm()
        dx_mm = g.dx * 1000.0
        off_mm = twin._window_offset_x_m * 1000.0
        i0 = max(0, min(int((x_mm - off_mm) / dx_mm), g.nx - 1))
        j0 = max(0, min(int(y_mm / dx_mm), g.ny - 1))
        free_surface.surface_height_at(
            g.phi, g.flags, g.surface_k_buf,
            i0, j0, twin.nz_solid, g.FLAG_GAS, g.nz,
        )
        k_surface = float(g.surface_k_buf[None])
        if k_surface < 1.0:
            k_surface = float(max(1, twin.nz_solid - 1))
        z_mm = (k_surface + 0.5) * dx_mm
        return x_mm, y_mm, z_mm

    def torch_marker_mm(self) -> tuple[float, float, float]:
        """Yellow tip glyph: contact tip above the plate, not path ``z=0``.

        When ``use_torch_z`` is on, robot TCP Z from the path is used.
        Otherwise tip = free surface + CTWD (clamped into the grid).
        """
        twin = self.twin
        g = twin.grid
        x_mm, y_mm, z_surf_mm = self.torch_surface_mm()
        if getattr(twin, "use_torch_z", False):
            _, _, z_path_mm = self.torch_mm()
            # Prefer robot Z when it is clearly above the substrate datum.
            if z_path_mm > z_surf_mm * 0.25:
                return x_mm, y_mm, z_path_mm
        dx_mm = g.dx * 1000.0
        tip_mm = z_surf_mm + float(getattr(twin, "ctwd_m", 0.015)) * 1000.0
        z_max_mm = (g.nz - 1.5) * dx_mm
        # Keep the tip visible above the plate even if CTWD exceeds the air gap.
        tip_mm = min(max(tip_mm, z_surf_mm + dx_mm), z_max_mm)
        return x_mm, y_mm, tip_mm


def _resolve_job_path(job_arg: str | None) -> str | None:
    if job_arg is None:
        return None
    p = resolve_project_path(job_arg)
    return str(p)


def create_session(
    *,
    job: str | None = None,
    preset: str | None = None,
    material: str | None = None,
) -> ViewerSession:
    """
    Load twin from job YAML (default) or preset-only demo.

    Default job: ``jobs/examples/bead_on_plate.yaml`` (calibrated + VOF).
    """
    from .. import WAAMTwin

    job_path = _resolve_job_path(job)
    if job_path is None and job is not None:
        raise FileNotFoundError(f"Job file not found: {job}")

    if job_path is None:
        default_job = PROJECT_ROOT / "jobs/examples/bead_on_plate.yaml"
        if default_job.exists():
            job_path = str(default_job)

    if job_path:
        twin = WAAMTwin.from_job(job_path, preset_override=preset)
        job_cfg = load_job_config(job_path)
        label = os.path.basename(job_path)
        if preset:
            label = f"{label} (preset={preset})"
    else:
        preset_name = preset or os.environ.get("WAAM_PRESET", "standard")
        mat = material or "materials/validated/ER70S-6.v1.yaml"
        twin = WAAMTwin.from_preset(
            preset_name,
            material=mat,
            enable_vof=True,
            enable_heat_loss=True,
            enable_csf_tension=True,
        )
        job_cfg = {}
        label = f"preset:{preset_name}"

    twin.reset()
    g = twin.grid

    travel = twin.travel_speed_m_s
    if travel <= 0:
        proc = job_cfg.get("process", {})
        travel = float(proc.get("travel_speed_mm_s", 5.0)) / 1000.0
        twin.travel_speed_m_s = travel

    x0 = 4 * g.dx
    y0 = (g.ny // 2) * g.dx
    z0 = 0.0
    waypoints = parse_torch_path(job_cfg) if job_cfg else []
    if len(waypoints) >= 2:
        driver = TorchPathDriver(waypoints, travel)
    elif len(waypoints) == 1:
        driver = None
        x0, y0, z0 = waypoints[0]
    else:
        driver = None

    session = ViewerSession(
        twin=twin,
        job_label=label,
        path_driver=driver,
        torch_x_m=x0 if not driver else 0.0,
        torch_y_m=y0 if not driver else 0.0,
        torch_z_m=z0 if not driver else 0.0,
        torch_spd_m_s=travel if travel > 0 else 0.005,
        x_min_m=4 * g.dx,
        x_max_m=(g.nx - 4) * g.dx,
    )
    session.reset_motion()
    return session
