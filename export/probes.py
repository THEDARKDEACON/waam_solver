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
# Sample points per launch: a probe uses 1 (point), +6 (thermocouple bead
# average), +6 (position tolerance) points.
MAX_POINTS = MAX_PROBES * 13
# Six points on the axes at a = r·sqrt(3/5) average a smooth field like the
# whole ball of radius r does (second order: f + r²/10 ∇²f).
_BALL_AXIS_FRACTION = math.sqrt(0.6)
_AXES6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


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


def _sample_every_ms(entries) -> float:
    """Job ``probes.sample_every_ms`` (dict form); 1 ms by default."""
    if isinstance(entries, dict) and entries.get("sample_every_ms") is not None:
        return max(0.0, float(entries["sample_every_ms"]))
    return 1.0


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
    # Sensor model (0 = off). A thermocouple reads the mean over its bead and
    # lags the surface temperature by its time constant; near the fusion line
    # an ideal point reads sharper peaks than any real sensor.
    bead_diameter_m: float = 0.0
    time_constant_s: float = 0.0
    # Placement uncertainty: also record min/max T at ±tolerance on each axis.
    position_tolerance_m: float = 0.0

    @property
    def has_sensor_model(self) -> bool:
        return self.bead_diameter_m > 0.0 or self.time_constant_s > 0.0

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
    # Record every sample_every_ms of simulated time (0 = every step). Each
    # sample reads the device, which stalls the step pipeline; thermocouple
    # loggers run at ≤ 1 kHz, so jobs default to 1 ms.
    sample_every_ms: float = 0.0
    _rows: list[dict[str, float | str]] = field(default_factory=list)
    _next_sample_ms: float = field(default=-1.0, repr=False)
    _tc_state: dict = field(default_factory=dict, repr=False)
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
        rec = cls(sample_every_ms=_sample_every_ms(entries))
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
            spec = rec.probes[-1]
            spec.bead_diameter_m = float(entry.get("bead_diameter_mm", 0.0)) / 1000.0
            spec.time_constant_s = float(entry.get("time_constant_ms", 0.0)) / 1000.0
            spec.position_tolerance_m = float(entry.get("position_tolerance_mm", 0.0)) / 1000.0
        return rec

    def to_job_block(self) -> dict:
        """The probes as a job ``probes:`` block (z_ref world, domain mm), so a
        probe set placed in the viewer can be pasted into a job file."""
        pts = []
        for p in self.probes:
            e: dict = {"name": p.name}
            if p.follow_torch:
                e["behind_mm"] = round(p.behind_m * 1000.0, 3)
                e["lateral_mm"] = round(p.lateral_m * 1000.0, 3)
            else:
                e["x_mm"] = round(p.x_m * 1000.0, 3)
                e["y_mm"] = round(p.y_m * 1000.0, 3)
            e["z_mm"] = round(p.z_m * 1000.0, 3)
            if p.bead_diameter_m > 0:
                e["bead_diameter_mm"] = round(p.bead_diameter_m * 1000.0, 3)
            if p.time_constant_s > 0:
                e["time_constant_ms"] = round(p.time_constant_s * 1000.0, 3)
            if p.position_tolerance_m > 0:
                e["position_tolerance_mm"] = round(p.position_tolerance_m * 1000.0, 3)
            pts.append(e)
        return {"z_ref": "world", "sample_every_ms": self.sample_every_ms, "points": pts}

    def record_step(self, twin: "WAAMTwin") -> None:
        if not self.probes:
            return
        g = twin.grid
        t_ms = twin._step_n * g.dt * 1000.0
        if self.sample_every_ms > 0.0:
            if t_ms + 1e-9 < self._next_sample_ms:
                return
            k = math.floor(t_ms / self.sample_every_ms + 1e-9) + 1
            self._next_sample_ms = k * self.sample_every_ms
        row: dict[str, float | str] = {
            "sim_time_ms": round(t_ms, 4),
            "step": twin._step_n,
            "arc_on": 1 if getattr(twin, "_arc_on", True) else 0,
        }
        # (probe, first point index, n tc points, n tolerance points)
        live: list[tuple[ProbeSpec, int, int, int]] = []
        pts: list[tuple[float, float, float]] = []
        inv = 1.0 / g.dx
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
            c = (
                (world[0] - twin._window_offset_x_m) * inv,
                (world[1] - twin._window_offset_y_m) * inv,
                (world[2] - twin._window_offset_z_m) * inv,
            )
            first = len(pts)
            pts.append(c)
            n_tc = 0
            if p.bead_diameter_m > 0.0:
                a = 0.5 * p.bead_diameter_m * _BALL_AXIS_FRACTION * inv
                pts.extend((c[0] + a * u[0], c[1] + a * u[1], c[2] + a * u[2]) for u in _AXES6)
                n_tc = 6
            n_tol = 0
            if p.position_tolerance_m > 0.0:
                a = p.position_tolerance_m * inv
                pts.extend((c[0] + a * u[0], c[1] + a * u[1], c[2] + a * u[2]) for u in _AXES6)
                n_tol = 6
            live.append((p, first, n_tc, n_tol))
        if live:
            vals = self._sample(twin, pts)
            for p, first, n_tc, n_tol in live:
                T_K, fl = vals[first]
                ok = T_K >= 0.0
                row[f"{p.name}_T_K"] = round(float(T_K), 2) if ok else math.nan
                row[f"{p.name}_f_l"] = round(float(fl), 4) if ok else math.nan
                if p.has_sensor_model:
                    row[f"{p.name}_T_tc_K"] = self._sensor(p, vals, first, n_tc, t_ms)
                if n_tol:
                    tv = [float(v[0]) for v in vals[first + 1 + n_tc:first + 1 + n_tc + n_tol] if v[0] >= 0.0]
                    row[f"{p.name}_T_lo_K"] = round(min(tv), 2) if tv else math.nan
                    row[f"{p.name}_T_hi_K"] = round(max(tv), 2) if tv else math.nan
        self._rows.append(row)

    def _sensor(self, p: ProbeSpec, vals, first: int, n_tc: int, t_ms: float) -> float:
        """Bead-averaged, first-order-lagged thermocouple reading [K]."""
        if n_tc:
            tv = [float(v[0]) for v in vals[first + 1:first + 1 + n_tc] if v[0] >= 0.0]
            T_in = sum(tv) / len(tv) if tv else (float(vals[first][0]) if vals[first][0] >= 0 else math.nan)
        else:
            T_in = float(vals[first][0]) if vals[first][0] >= 0.0 else math.nan
        prev = self._tc_state.get(p.name)
        if not math.isfinite(T_in):
            return round(prev[1], 2) if prev else math.nan
        if prev is None or p.time_constant_s <= 0.0 or not math.isfinite(prev[1]):
            out = T_in
        else:
            dt = max(0.0, (t_ms - prev[0]) / 1000.0)
            out = prev[1] + (T_in - prev[1]) * (1.0 - math.exp(-dt / p.time_constant_s))
        self._tc_state[p.name] = (t_ms, out)
        return round(out, 2)

    def series(self, name: str) -> tuple[list[float], list[float]]:
        """(sim_time_ms, T_K) of one probe: the thermocouple model when the
        probe has one, else the point value. NaN samples are dropped."""
        spec = next((q for q in self.probes if q.name == name), None)
        key = f"{name}_T_tc_K" if spec is not None and spec.has_sensor_model else f"{name}_T_K"
        t, T = [], []
        for r in self._rows:
            v = r.get(key)
            if v is None:
                continue
            v = float(v)
            if math.isfinite(v):
                t.append(float(r["sim_time_ms"]))
                T.append(v)
        return t, T

    def summary(self, T_solidus_K: float) -> list[dict]:
        """Weld thermal-cycle metrics per probe (see probe_metrics)."""
        from .probe_metrics import thermal_cycle_metrics
        out = []
        for p in self.probes:
            t, T = self.series(p.name)
            m = thermal_cycle_metrics(t, T, T_solidus_K)
            out.append({"name": p.name, **m})
        return out

    def write_summary_csv(self, path: str | pathlib.Path, T_solidus_K: float) -> None:
        rows = self.summary(T_solidus_K)
        if not rows:
            return
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _sample(self, twin: "WAAMTwin", cells: list[tuple[float, float, float]]):
        """One kernel launch and one device read for every live probe."""
        import numpy as np

        g = twin.grid
        n = len(cells)
        if n > MAX_POINTS:
            raise ValueError(f"{n} probe sample points exceed MAX_POINTS={MAX_POINTS}")
        if self._dev is None or self._dev[0] is not g:
            self._dev = (
                g,
                ti.field(dtype=ti.f32, shape=(MAX_POINTS, 3)),
                ti.field(dtype=ti.f32, shape=(MAX_POINTS, 2)),
                None,
            )
        grid, pos_f, out_f, uploaded = self._dev
        pos = np.zeros((MAX_POINTS, 3), dtype=np.float32)
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
            if p.has_sensor_model:
                names.append(f"{p.name}_T_tc_K")
            if p.position_tolerance_m > 0.0:
                names.append(f"{p.name}_T_lo_K")
                names.append(f"{p.name}_T_hi_K")
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
        names = tuple(self._fieldnames())
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
        self._next_sample_ms = -1.0
        self._tc_state.clear()
