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
    """A probe fixed at a WORLD position (like a physical thermocouple).

    World coordinates are stored and re-resolved to grid indices at every
    sample so the probe stays at the same physical location when the moving
    window shifts. (Previously indices were resolved once at creation: after
    a window shift the "thermocouple" silently measured a different point.)
    """

    name: str
    x_m: float
    y_m: float
    z_m: float

    def resolve(self, twin: "WAAMTwin") -> tuple[int, int, int] | None:
        g = twin.grid
        i = int((self.x_m - twin._window_offset_x_m) / g.dx)
        j = int(self.y_m / g.dx)
        k = int(self.z_m / g.dx)
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
            y_m=(j + 0.5) * g.dx,
            z_m=(k + 0.5) * g.dx,
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

    @classmethod
    def from_job_list(cls, entries: list[dict], twin: "WAAMTwin") -> "ProbeRecorder":
        rec = cls()
        for entry in entries:
            name = str(entry.get("name", f"probe_{len(rec.probes)}"))
            if "i" in entry and "j" in entry and "k" in entry:
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
            fieldnames.append(f"{p.name}_T_K")
            fieldnames.append(f"{p.name}_f_l")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def clear(self) -> None:
        self._rows.clear()
