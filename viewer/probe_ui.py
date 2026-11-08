"""Probe placement and display for the live viewer.

Probes are placed on metal you can see: right-click a point (or ``I`` for
the screen centre) and a ray is marched through the grid to the first metal
cell, honouring the section plane, so a click on the cut face puts the probe
inside the part. ``U`` puts one 1 mm under the metal surface below the torch.

Each new probe is either fixed in the world (a thermocouple) or rides with the
torch at the same distance behind and beside it (``L`` toggles the mode).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

from waam_twin.compiler import ti

C0 = 273.15
UNDER_SURFACE_MM = 1.0


class ProbeUI:
    def __init__(self, twin, out_dir: pathlib.Path) -> None:
        from waam_twin.export.probes import ProbeRecorder

        self.twin = twin
        self.out_dir = pathlib.Path(out_dir)
        if twin.probe_recorder is None:
            twin.probe_recorder = ProbeRecorder(sample_every_ms=1.0)
        self.rec = twin.probe_recorder
        self.selected = 0
        self.follow_torch = False
        self._hit = ti.field(dtype=ti.f32, shape=4)
        self._summary: list[dict] = []
        self._n_saved = -1

    # ------------------------------------------------------------ placement
    def pick(self, origin_mm, dir_mm, cut_cells) -> np.ndarray | None:
        """World mm of the first metal (kept side of the cut) along a ray."""
        from .extract import ray_pick_metal

        twin, g = self.twin, self.twin.grid
        dx_mm = g.dx * 1000.0
        off = np.array(twin.window_offset_m(), dtype=np.float64) * 1000.0
        o = (np.asarray(origin_mm, dtype=np.float64) - off) / dx_mm
        d = np.asarray(dir_mm, dtype=np.float64)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        on, cnx, cny, cnz, cd = cut_cells
        ray_pick_metal(
            g.phi, g.flags, self._hit,
            float(o[0]), float(o[1]), float(o[2]),
            float(d[0]), float(d[1]), float(d[2]),
            float(np.linalg.norm(o)) + 4.0 * float(g.nx + g.ny + g.nz),
            g.FLAG_GAS, g.nx, g.ny, g.nz,
            on, cnx, cny, cnz, cd,
        )
        h = self._hit.to_numpy()
        if h[0] < 0.5:
            return None
        # Half a cell further in, so the probe sits in the metal, not on its skin.
        p_cells = h[1:4] + 0.5 * d
        return p_cells * dx_mm + off

    def add(self, world_mm, name: str | None = None) -> str:
        twin = self.twin
        rec = self.rec
        name = name or self._next_name()
        x, y, z = (float(v) for v in world_mm)
        if self.follow_torch and twin._last_torch_pos_m is not None:
            tx, ty, _ = (v * 1000.0 for v in twin._last_torch_pos_m)
            hx, hy, _ = twin._torch_dir_xyz
            hn = math.hypot(hx, hy) or 1.0
            hx, hy = hx / hn, hy / hn
            vx, vy = x - tx, y - ty
            behind = -(hx * vx + hy * vy)
            lateral = -hy * vx + hx * vy
            rec.add_torch_relative(behind, z, twin, name, lateral_mm=lateral)
        else:
            rec.add_world_mm(x, y, z, twin, name)
        self.selected = len(rec.probes) - 1
        self.save()
        return name

    def add_under_torch(self, torch_surface_mm) -> str:
        x, y, z = torch_surface_mm
        return self.add((x, y, z - UNDER_SURFACE_MM))

    def remove_selected(self) -> str | None:
        if not self.rec.probes:
            return None
        name = self.rec.probes.pop(self.selected).name
        self.selected = max(0, min(self.selected, len(self.rec.probes) - 1))
        self.save()
        return name

    def cycle(self) -> None:
        if self.rec.probes:
            self.selected = (self.selected + 1) % len(self.rec.probes)

    def _next_name(self) -> str:
        used = {p.name for p in self.rec.probes}
        i = 1
        while f"P{i}" in used:
            i += 1
        return f"P{i}"

    def save(self) -> pathlib.Path | None:
        """probes.yaml in the output folder: a job ``probes:`` block."""
        import yaml

        if not self.rec.probes and self._n_saved <= 0:
            return None
        path = self.out_dir / "probes.yaml"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"probes": self.rec.to_job_block()}, sort_keys=False), encoding="utf-8")
        self._n_saved = len(self.rec.probes)
        return path

    # ------------------------------------------------------------ display
    def markers(self) -> list[tuple[np.ndarray, list[str], tuple[float, float, float]]]:
        """(points world mm, labels, colour) for fixed and torch-riding probes."""
        groups = {False: ([], []), True: ([], [])}
        for i, p in enumerate(self.rec.probes):
            w = p.world_m(self.twin)
            if w is None:
                continue
            pts, txt = groups[p.follow_torch]
            pts.append([v * 1000.0 for v in w])
            txt.append((">" if i == self.selected else "") + p.name)
        out = []
        for follow, colour in ((False, (1.0, 0.2, 0.9)), (True, (0.2, 0.9, 1.0))):
            pts, txt = groups[follow]
            if pts:
                out.append((np.array(pts), txt, colour))
        return out

    def refresh_summary(self, T_solidus_K: float) -> None:
        self._summary = self.rec.summary(T_solidus_K)

    def hud_lines(self, max_lines: int = 6) -> list[str]:
        rec = self.rec
        mode = "follows torch" if self.follow_torch else "fixed"
        head = f"Probes : {len(rec.probes)}  new={mode} (L)  RMB/I/U add  J next  Del remove"
        lines = [head]
        by_name = {m["name"]: m for m in self._summary}
        n = len(rec.probes)
        if n == 0:
            return lines
        lo = max(0, min(self.selected - max_lines // 2, n - max_lines))
        for i in range(lo, min(n, lo + max_lines)):
            p = rec.probes[i]
            m = by_name.get(p.name, {})
            tag = "tc" if p.has_sensor_model else "pt"
            lines.append(
                f" {'>' if i == self.selected else ' '}{p.name:<6} {tag} "
                f"T {_c(m.get('T_now_C'))}  peak {_c(m.get('peak_C'))}  "
                f"t8/5 {_s(m.get('t85_s'))}  "
                f"{'rides torch' if p.follow_torch else 'fixed'}"
            )
        return lines

    def spark(self, width: int = 32) -> tuple[str, list[float]]:
        """(name, last T values in K) of the selected probe for the graph."""
        if not self.rec.probes:
            return "", []
        name = self.rec.probes[self.selected].name
        _, T = self.rec.series(name)
        return name, T[-width * 8:][::8] if len(T) > width * 8 else T[-width:]


def _c(v) -> str:
    return "  —  " if v is None or not math.isfinite(v) else f"{v:5.0f}°C"


def _s(v) -> str:
    return " — " if v is None or not math.isfinite(v) else f"{v:.2f}s"
