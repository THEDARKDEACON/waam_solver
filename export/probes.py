"""Thermocouple-style probe recording for T(t) research export."""

from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from waam_twin.compiler import ti

if TYPE_CHECKING:
    from ..twin import WAAMTwin

MAX_PROBES = 64


@ti.kernel
def _sample_probes(
    T: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    pos: ti.template(),
    out: ti.template(),
    n: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Trilinear T and f_l at positions in cell units; gas corners carry no weight.

    out[p, 0] < 0 when all eight corners are gas.
    """
    for p in range(n):
        fx = pos[p, 0] - 0.5
        fy = pos[p, 1] - 0.5
        fz = pos[p, 2] - 0.5
        i0 = ti.cast(ti.floor(fx), ti.i32)
        j0 = ti.cast(ti.floor(fy), ti.i32)
        k0 = ti.cast(ti.floor(fz), ti.i32)
        wx = fx - ti.cast(i0, ti.f32)
        wy = fy - ti.cast(j0, ti.f32)
        wz = fz - ti.cast(k0, ti.f32)
        w_sum = 0.0
        T_sum = 0.0
        fl_sum = 0.0
        for c in ti.static(range(8)):
            ii = ti.max(0, ti.min(i0 + (c & 1), nx - 1))
            jj = ti.max(0, ti.min(j0 + ((c >> 1) & 1), ny - 1))
            kk = ti.max(0, ti.min(k0 + ((c >> 2) & 1), nz - 1))
            w = (wx if c & 1 else 1.0 - wx) * (wy if (c >> 1) & 1 else 1.0 - wy) * (
                wz if (c >> 2) & 1 else 1.0 - wz
            )
            if flags[ii, jj, kk] != FLAG_GAS:
                w_sum += w
                T_sum += w * T[ii, jj, kk]
                fl_sum += w * f_l[ii, jj, kk]
        out[p, 0] = -1.0
        out[p, 1] = 0.0
        if w_sum > 1e-6:
            out[p, 0] = T_sum / w_sum
            out[p, 1] = fl_sum / w_sum

_Z_REF_ALIASES = {
    "plate_bottom": "plate_bottom",
    "bottom": "plate_bottom",
    "plate_face": "plate_face",
    "face": "plate_face",
    "world": "world",
    "domain": "world",
}


def normalize_z_ref(z_ref: str) -> str:
    """Map job spelling to plate_bottom | plate_face | world."""
    key = str(z_ref).strip().lower().replace("-", "_")
    if key not in _Z_REF_ALIASES:
        raise ValueError(
            f"Unknown z_ref={z_ref!r}. Use plate_bottom, plate_face, or world."
        )
    return _Z_REF_ALIASES[key]


def resolve_probe_z_world_mm(twin: "WAAMTwin", z_mm: float, z_ref: str) -> float:
    """Convert a job probe height to domain-world millimetres (k=0 floor).

    ``plate_bottom`` — z=0 at the metal band bottom (``plate_k0``).
    ``plate_face``   — z=0 at the plate-top face (``nz_solid``); negative into metal.
    ``world``        — z already in domain coordinates.
    """
    ref = normalize_z_ref(z_ref)
    z = float(z_mm)
    if ref == "world":
        return z
    dx_mm = float(twin.grid.dx) * 1000.0
    if ref == "plate_bottom":
        k0 = int(getattr(twin, "plate_k0", 0))
        return k0 * dx_mm + z
    # plate_face
    face = int(getattr(twin, "nz_solid", 0))
    return face * dx_mm + z


def _coerce_probes_config(
    entries: list[dict] | dict,
) -> tuple[str, list[dict]]:
    """Return (default_z_ref, point dicts) from list or {z_ref, points} form."""
    if isinstance(entries, dict):
        default = str(entries.get("z_ref", "plate_bottom"))
        points = entries.get("points")
        if points is None:
            # Allow legacy key name
            points = entries.get("probes")
        if points is None:
            raise ValueError(
                "probes dict must include a 'points' list "
                "(and optional section-level z_ref)."
            )
        if not isinstance(points, list):
            raise ValueError("probes.points must be a list of probe entries.")
        return default, points
    if isinstance(entries, list):
        return "plate_bottom", entries
    raise TypeError(f"probes must be a list or dict, got {type(entries).__name__}")


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
    _flushed: int = 0
    _flush_names: tuple[str, ...] = ()
    _dev: tuple | None = field(default=None, repr=False)

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
    def from_job_list(
        cls,
        entries: list[dict] | dict,
        twin: "WAAMTwin",
    ) -> "ProbeRecorder":
        """Build probes from a job ``probes`` block; store world-frame positions.

        Job forms::

            probes:
              z_ref: plate_bottom   # or plate_face | world  (section default)
              points:
                - name: tc1
                  x_mm: 30
                  y_mm: 30
                  z_mm: 7.0
                  # z_ref: plate_face   # optional per-probe override

            # Legacy list form (default z_ref=plate_bottom):
            probes:
              - name: tc1
                x_mm: 30
                y_mm: 30
                z_mm: 7.0

        Legacy ``world_z: true`` on an entry maps to ``z_ref: world``.
        Grid-index probes (i/j/k) skip z_ref (already domain cells).
        """
        rec = cls()
        default_z_ref, points = _coerce_probes_config(entries)
        for entry in points:
            name = str(entry.get("name", f"probe_{len(rec.probes)}"))
            if "i" in entry and "j" in entry and "k" in entry:
                rec.add_grid(int(entry["i"]), int(entry["j"]), int(entry["k"]), twin, name)
                continue
            if "z_ref" in entry:
                z_ref = str(entry["z_ref"])
            elif bool(entry.get("world_z", False)):
                z_ref = "world"
            else:
                z_ref = default_z_ref
            z_world_mm = resolve_probe_z_world_mm(
                twin, float(entry.get("z_mm", 0.0)), z_ref
            )
            if "behind_mm" in entry:
                rec.add_torch_relative(
                    float(entry["behind_mm"]),
                    z_world_mm,
                    twin,
                    name,
                    lateral_mm=float(entry.get("lateral_mm", 0.0)),
                )
            else:
                rec.add_world_mm(
                    float(entry.get("x_mm", 0)),
                    float(entry.get("y_mm", 0)),
                    z_world_mm,
                    twin,
                    name,
                )
        return rec

    def record_step(self, twin: "WAAMTwin") -> None:
        if not self.probes:
            return
        g = twin.grid
        t_ms = twin._step_n * g.dt * 1000.0
        row: dict[str, float | str] = {
            "sim_time_ms": round(t_ms, 4),
            "step": twin._step_n,
            "arc_on": 1 if getattr(twin, "_arc_on", True) else 0,
        }
        live: list[tuple[str, tuple[float, float, float]]] = []
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
            if world is None or p.resolve(twin) is None:
                row[f"{p.name}_T_K"] = math.nan
                row[f"{p.name}_f_l"] = math.nan
                continue
            live.append((p.name, (
                (world[0] - twin._window_offset_x_m) / g.dx,
                (world[1] - twin._window_offset_y_m) / g.dx,
                (world[2] - twin._window_offset_z_m) / g.dx,
            )))
        if live:
            vals = self._sample(twin, [c for _, c in live])
            for (name, _), (T_K, fl) in zip(live, vals):
                ok = T_K >= 0.0
                row[f"{name}_T_K"] = round(float(T_K), 2) if ok else math.nan
                row[f"{name}_f_l"] = round(float(fl), 4) if ok else math.nan
        self._rows.append(row)

    def _sample(self, twin: "WAAMTwin", cells: list[tuple[float, float, float]]):
        """One kernel launch and one device read for every live probe."""
        import numpy as np

        g = twin.grid
        n = len(cells)
        if n > MAX_PROBES:
            raise ValueError(f"{n} probes exceeds MAX_PROBES={MAX_PROBES}")
        if self._dev is None or self._dev[0] is not g:
            self._dev = (
                g,
                ti.field(dtype=ti.f32, shape=(MAX_PROBES, 3)),
                ti.field(dtype=ti.f32, shape=(MAX_PROBES, 2)),
                None,
            )
        grid, pos_f, out_f, uploaded = self._dev
        pos = np.zeros((MAX_PROBES, 3), dtype=np.float32)
        pos[:n] = cells
        if uploaded is None or not np.array_equal(uploaded, pos):
            pos_f.from_numpy(pos)
            self._dev = (grid, pos_f, out_f, pos)
        _sample_probes(
            g.T, g.f_l, g.flags, pos_f, out_f, n, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        return out_f.to_numpy()[:n]

    def _fieldnames(self) -> list[str]:
        names = ["sim_time_ms", "step", "arc_on"]
        for p in self.probes:
            names.append(f"{p.name}_x_mm")
            names.append(f"{p.name}_y_mm")
            names.append(f"{p.name}_z_mm")
            names.append(f"{p.name}_T_K")
            names.append(f"{p.name}_f_l")
        return names

    def write_csv(self, path: str | pathlib.Path) -> None:
        if not self._rows:
            return
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    def flush_csv(self, path: str | pathlib.Path) -> int:
        """Append rows not yet on disk. The file keeps growing after the arc goes off."""
        if not self.probes or not self._rows:
            return 0
        path = pathlib.Path(path)
        names = tuple(p.name for p in self.probes)
        if names != self._flush_names:
            self._flushed = 0
            self._flush_names = names
            if path.exists():
                path.unlink()
        if self._flushed >= len(self._rows):
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        new = self._rows[self._flushed :]
        write_header = (not path.exists()) or path.stat().st_size == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames(), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(new)
        self._flushed = len(self._rows)
        return len(new)

    def clear(self) -> None:
        self._rows.clear()
        self._flushed = 0
