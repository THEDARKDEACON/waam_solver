"""Thermocouple-style probe recording for T(t) research export."""

from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin


@dataclass
class ProbeSpec:
    """A sample point, either fixed in the world or riding with the torch.

    Fixed probes store a world position (like a physical thermocouple) and
    re-resolve it to grid indices at every sample, so the point stays put
    when the moving window shifts.

    Torch-relative probes store ``behind_m`` / ``lateral_m`` and an absolute
    sample height ``z_m``. Each sample rebuilds the world XY from the current
    torch position and horizontal travel direction, then resolves that point
    the same way. Positive ``behind_m`` is opposite travel. Positive
    ``lateral_m`` is to the left of travel.
    """

    name: str
    x_m: float
    y_m: float
    z_m: float
    follow_torch: bool = False
    behind_m: float = 0.0
    lateral_m: float = 0.0

    def world_m(self, twin: "WAAMTwin") -> tuple[float, float, float] | None:
        if not self.follow_torch:
            return (self.x_m, self.y_m, self.z_m)
        pos = getattr(twin, "_last_torch_pos_m", None)
        if pos is None:
            return None
        tx, ty, _tz = pos
        dx, dy, _dz = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
        horiz = math.hypot(float(dx), float(dy))
        if horiz > 1e-12:
            hx, hy = float(dx) / horiz, float(dy) / horiz
        else:
            hx, hy = 0.0, 0.0
        x = tx - hx * self.behind_m - hy * self.lateral_m
        y = ty - hy * self.behind_m + hx * self.lateral_m
        return (x, y, self.z_m)

    def resolve(self, twin: "WAAMTwin") -> tuple[int, int, int] | None:
        world = self.world_m(twin)
        if world is None:
            return None
        x_m, y_m, z_m = world
        g = twin.grid
        i = int((x_m - twin._window_offset_x_m) / g.dx)
        j = int((y_m - twin._window_offset_y_m) / g.dx)
        k = int((z_m - twin._window_offset_z_m) / g.dx)
        if not (0 <= i < g.nx and 0 <= j < g.ny and 0 <= k < g.nz):
            return None  # probe left the simulation window
        return i, j, k


@dataclass
class ProbeRecorder:
    """Record temperature history at world-fixed probe locations."""

    probes: list[ProbeSpec] = field(default_factory=list)
    _rows: list[dict[str, float | str]] = field(default_factory=list)

    def add_grid(
        self, i: int, j: int, k: int, twin: "WAAMTwin", name: str | None = None
    ) -> None:
        """Add a probe by CURRENT grid index (converted to world position)."""
        g = twin.grid
        label = name or f"probe_{len(self.probes)}"
        self.probes.append(ProbeSpec(
            name=label,
            x_m=(i + 0.5) * g.dx + twin._window_offset_x_m,
            y_m=(j + 0.5) * g.dx + twin._window_offset_y_m,
            z_m=(k + 0.5) * g.dx + twin._window_offset_z_m,
        ))

    def add_world_mm(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        twin: "WAAMTwin",
        name: str | None = None,
    ) -> None:
        label = name or f"probe_{len(self.probes)}"
        self.probes.append(ProbeSpec(
            name=label, x_m=x_mm / 1000.0, y_m=y_mm / 1000.0, z_m=z_mm / 1000.0,
        ))

    def add_torch_relative(
        self,
        behind_mm: float,
        z_mm: float,
        twin: "WAAMTwin",
        name: str | None = None,
        lateral_mm: float = 0.0,
    ) -> None:
        """Sample ``behind_mm`` opposite travel, at a fixed domain height.

        ``twin`` is accepted for the same call shape as the other add methods.
        The pose is read when the probe is sampled, not when it is created.
        """
        label = name or f"probe_{len(self.probes)}"
        self.probes.append(ProbeSpec(
            name=label,
            x_m=0.0,
            y_m=0.0,
            z_m=float(z_mm) / 1000.0,
            follow_torch=True,
            behind_m=float(behind_mm) / 1000.0,
            lateral_m=float(lateral_mm) / 1000.0,
        ))

    @classmethod
    def from_job_list(cls, entries: list[dict], twin: "WAAMTwin") -> "ProbeRecorder":
        rec = cls()
        for entry in entries:
            name = str(entry.get("name", f"probe_{len(rec.probes)}"))
            if "behind_mm" in entry:
                rec.add_torch_relative(
                    float(entry["behind_mm"]),
                    float(entry.get("z_mm", 0.0)),
                    twin,
                    name,
                    lateral_mm=float(entry.get("lateral_mm", 0.0)),
                )
            elif "i" in entry and "j" in entry and "k" in entry:
                rec.add_grid(int(entry["i"]), int(entry["j"]), int(entry["k"]), twin, name)
            else:
                rec.add_world_mm(
                    float(entry.get("x_mm", 0)),
                    float(entry.get("y_mm", 0)),
                    float(entry.get("z_mm", 0)),
                    twin,
                    name,
                )
        return rec

    def record_step(self, twin: "WAAMTwin") -> None:
        if not self.probes:
            return
        g = twin.grid
        t_ms = twin._step_n * g.dt * 1000.0
        row: dict[str, float | str] = {"sim_time_ms": round(t_ms, 4), "step": twin._step_n}
        for p in self.probes:
            world = p.world_m(twin)
            if world is None:
                row[f"{p.name}_x_mm"] = math.nan
                row[f"{p.name}_y_mm"] = math.nan
                row[f"{p.name}_z_mm"] = math.nan
            else:
                row[f"{p.name}_x_mm"] = round(world[0] * 1000.0, 4)
                row[f"{p.name}_y_mm"] = round(world[1] * 1000.0, 4)
                row[f"{p.name}_z_mm"] = round(world[2] * 1000.0, 4)
            idx = p.resolve(twin)
            if idx is None:
                row[f"{p.name}_T_K"] = math.nan
                row[f"{p.name}_f_l"] = math.nan
                continue
            i, j, k = idx
            # Scalar field reads — avoids copying the full T/f_l volumes to
            # host every step (previously two full to_numpy() per sample).
            row[f"{p.name}_T_K"] = round(float(g.T[i, j, k]), 2)
            row[f"{p.name}_f_l"] = round(float(g.f_l[i, j, k]), 4)
        self._rows.append(row)

    def write_csv(self, path: str | pathlib.Path) -> None:
        if not self._rows:
            return
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = ["sim_time_ms", "step"]
        for p in self.probes:
            fieldnames.append(f"{p.name}_x_mm")
            fieldnames.append(f"{p.name}_y_mm")
            fieldnames.append(f"{p.name}_z_mm")
            fieldnames.append(f"{p.name}_T_K")
            fieldnames.append(f"{p.name}_f_l")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def clear(self) -> None:
        self._rows.clear()
