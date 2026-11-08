"""Section (cut) plane for the live viewer.

Everything on one side of the plane is kept and drawn; the cut face is
capped and coloured, like a CAD section view or a macro-etch.

Axis-aligned cuts (X / Y / Z keys) have two anchors:
  torch  X/Y: the plane rides with the torch at ``offset_mm`` along the axis
         (negative = behind the arc when travel is along that axis).
         Z: the plane sits ``offset_mm`` above the plate face.
  fixed  the plane stays at world coordinate ``pos_mm``.
A free plane (any normal, from the mouse plane widget) is always fixed.

The kept side faces away from the camera by default, so the cut face is the
side you look at; ``flip`` keeps the other side instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXIS_NAMES = ("X", "Y", "Z")
FREE = 3


@dataclass
class CutPlane:
    axis: int | None = None  # 0, 1, 2, FREE, or None (off)
    anchor: str = "torch"
    offset_mm: float = 0.0
    pos_mm: float = 0.0
    flip: bool = False
    free_normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    free_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    version: int = 0  # bumped on every change (mesh rebuild key)

    @property
    def active(self) -> bool:
        return self.axis is not None

    def _touch(self) -> None:
        self.version += 1

    def select(self, axis: int, travel_dir: tuple[float, float, float], behind_mm: float = 3.0,
               z_offset_mm: float = 0.0) -> None:
        """X/Y/Z key: switch to that axis, or off when it is already active.

        Z starts ``z_offset_mm`` from the plate face: pass about −½ cell, the
        centre of the top plate layer. Exactly on the face φ is 0.5 and the
        filled cut face breaks up."""
        if self.axis == axis:
            self.axis = None
            self._touch()
            return
        self.axis = axis
        self.anchor = "torch"
        self.flip = False
        self.offset_mm = z_offset_mm if axis == 2 else 0.0
        if axis in (0, 1):
            d = np.asarray(travel_dir, dtype=np.float64)
            if abs(d[axis]) >= max(abs(d[1 - axis]), 1e-9):
                # Plane across the travel direction: behind the arc, where
                # the bead has just been laid down.
                self.offset_mm = -behind_mm * float(np.sign(d[axis]) or 1.0)
        self._touch()

    def set_free(self, normal, origin) -> None:
        n = np.asarray(normal, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            return
        self.axis = FREE
        self.anchor = "fixed"
        self.free_normal = n / nn
        self.free_origin = np.asarray(origin, dtype=np.float64)
        self._touch()

    def position_mm(self, torch_mm, face_mm: float) -> float:
        """World coordinate of an axis-aligned plane along its axis."""
        if self.anchor == "fixed":
            return self.pos_mm
        if self.axis == 2:
            return face_mm + self.offset_mm
        return float(torch_mm[self.axis]) + self.offset_mm

    def nudge(self, delta_mm: float) -> None:
        if self.axis is None:
            return
        if self.axis == FREE:
            self.free_origin = self.free_origin + delta_mm * self.free_normal
        elif self.anchor == "fixed":
            self.pos_mm += delta_mm
        else:
            self.offset_mm += delta_mm
        self._touch()

    def toggle_anchor(self, torch_mm, face_mm: float) -> None:
        """Switch torch ↔ fixed without moving the plane."""
        if self.axis is None or self.axis == FREE:
            return
        here = self.position_mm(torch_mm, face_mm)
        if self.anchor == "torch":
            self.anchor = "fixed"
            self.pos_mm = here
        else:
            self.anchor = "torch"
            ref = face_mm if self.axis == 2 else float(torch_mm[self.axis])
            self.offset_mm = here - ref
        self._touch()

    def toggle_flip(self) -> None:
        self.flip = not self.flip
        self._touch()

    def plane(self, torch_mm, face_mm: float, camera_mm) -> tuple[np.ndarray, np.ndarray] | None:
        """(unit normal pointing into the KEPT side, origin) in world mm."""
        if self.axis is None:
            return None
        if self.axis == FREE:
            n = self.free_normal.copy()
            o = self.free_origin.copy()
        else:
            n = np.zeros(3)
            n[self.axis] = 1.0
            o = np.asarray(torch_mm, dtype=np.float64).copy()
            o[self.axis] = self.position_mm(torch_mm, face_mm)
            if self.axis == 2:
                o[:2] = np.asarray(torch_mm, dtype=np.float64)[:2]
        # Keep the side away from the camera so the cut face is visible.
        if float(np.dot(n, np.asarray(camera_mm, dtype=np.float64) - o)) > 0.0:
            n = -n
        if self.flip:
            n = -n
        return n, o

    def label(self, torch_mm, face_mm: float, travel_dir) -> str:
        if self.axis is None:
            return "off  (X/Y/Z)"
        if self.axis == FREE:
            n, o = self.free_normal, self.free_origin
            return (f"free  n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f}) through "
                    f"({o[0]:.1f},{o[1]:.1f},{o[2]:.1f}) mm")
        name = AXIS_NAMES[self.axis]
        p = self.position_mm(torch_mm, face_mm)
        if self.anchor == "fixed":
            where = "fixed"
        elif self.axis == 2:
            where = f"plate face {self.offset_mm:+.1f}"
        else:
            d = float(np.asarray(travel_dir)[self.axis])
            if abs(d) > 0.5 and self.offset_mm * d < 0:
                where = f"{abs(self.offset_mm):.1f} behind arc"
            elif abs(d) > 0.5 and self.offset_mm * d > 0:
                where = f"{abs(self.offset_mm):.1f} ahead of arc"
            else:
                where = f"torch {self.offset_mm:+.1f}"
        side = "flipped" if self.flip else "auto side"
        return f"{name} @ {p:.1f} mm ({where}, {side})"

    def cell_plane(self, n: np.ndarray, o_mm: np.ndarray, offset_mm, dx_mm: float) -> tuple[float, float, float, float]:
        """(nx, ny, nz, d) in window-local cell units: keep cells with
        n·(centre) − d ≥ 0, centre = index + 0.5."""
        o_cells = (np.asarray(o_mm, dtype=np.float64) - np.asarray(offset_mm, dtype=np.float64)) / dx_mm
        return float(n[0]), float(n[1]), float(n[2]), float(np.dot(n, o_cells))
