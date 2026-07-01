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

    @property
    def uses_path(self) -> bool:
        return self.path_driver is not None

    def reset_motion(self) -> None:
        """Reset simulation and torch position to job start."""
        self.twin.reset()
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
            if self.path_driver:
                sim_t = self.twin._step_n * g.dt
                x, y, z = self.path_driver.position_at_time(sim_t)
                x, y = clamp_torch_to_domain(x, y, g.nx, g.ny, g.dx)
                self.torch_x_m, self.torch_y_m, self.torch_z_m = x, y, z
            else:
                self.torch_x_m += self._bounce_dir * self.torch_spd_m_s * g.dt
                if self.torch_x_m >= self.x_max_m or self.torch_x_m <= self.x_min_m:
                    self._bounce_dir *= -1
                    self.torch_x_m = max(self.x_min_m, min(self.torch_x_m, self.x_max_m))
            self.twin.step(self.torch_x_m, self.torch_y_m, is_welding=is_welding)

    def offset_x_mm(self) -> float:
        return self.twin._window_offset_x_m * 1000.0

    def torch_mm(self) -> tuple[float, float, float]:
        x, y, z = self.torch_position_now()
        off = self.offset_x_mm()
        g = self.twin.grid
        return x * 1000.0 + off, y * 1000.0, z * 1000.0


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
