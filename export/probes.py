"""Thermocouple-style probe recording for T(t) research export."""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..twin import WAAMTwin


@dataclass
class ProbeSpec:
    name: str
    i: int
    j: int
    k: int


@dataclass
class ProbeRecorder:
    """Record temperature history at fixed grid indices."""

    probes: list[ProbeSpec] = field(default_factory=list)
    _rows: list[dict[str, float | str]] = field(default_factory=list)

    def add_grid(self, i: int, j: int, k: int, name: str | None = None) -> None:
        label = name or f"probe_{len(self.probes)}"
        self.probes.append(ProbeSpec(name=label, i=i, j=j, k=k))

    def add_world_mm(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        twin: "WAAMTwin",
        name: str | None = None,
    ) -> None:
        g = twin.grid
        ox = twin._window_offset_x_m * 1000.0
        i = int((x_mm - ox) / (g.dx * 1000.0))
        j = int(y_mm / (g.dx * 1000.0))
        k = int(z_mm / (g.dx * 1000.0))
        i = max(0, min(g.nx - 1, i))
        j = max(0, min(g.ny - 1, j))
        k = max(0, min(g.nz - 1, k))
        self.add_grid(i, j, k, name)

    @classmethod
    def from_job_list(cls, entries: list[dict], twin: "WAAMTwin") -> "ProbeRecorder":
        rec = cls()
        for entry in entries:
            name = str(entry.get("name", f"probe_{len(rec.probes)}"))
            if "i" in entry and "j" in entry and "k" in entry:
                rec.add_grid(int(entry["i"]), int(entry["j"]), int(entry["k"]), name)
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
        T_np = g.T.to_numpy()
        fl_np = g.f_l.to_numpy()
        t_ms = twin._step_n * g.dt * 1000.0
        row: dict[str, float | str] = {"sim_time_ms": round(t_ms, 4), "step": twin._step_n}
        for p in self.probes:
            row[f"{p.name}_T_K"] = round(float(T_np[p.i, p.j, p.k]), 2)
            row[f"{p.name}_f_l"] = round(float(fl_np[p.i, p.j, p.k]), 4)
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
