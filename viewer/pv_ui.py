"""PyVista stand-in for Taichi GGUI (removed in Quadrants).

Mirrors the subset of ``ti.ui`` the melt-pool viewer uses: a window, a
Z-up FPS camera, particle/line drawing, and a text HUD. Requires the
``export`` extra (PyVista).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PRESS = "press"
SPACE = " "
ESCAPE = "escape"
LMB = "lmb"

# PyVista default bindings that collide with the GGUI camera / viewer keys.
_PYVISTA_KEYS_TO_CLEAR = ("q", "b", "v", "C", "plus", "minus", "Up", "Down")


def _waam_interactor_style():
    """Style that does not treat Q/E as quit (those are camera down/up)."""
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

    class WaamInteractorStyle(vtkInteractorStyleUser):
        def OnChar(self):  # noqa: N802
            return

    return WaamInteractorStyle()


def normalize_vtk_key(sym: str, code: str) -> str:
    """Map VTK KeySym/KeyCode onto the strings the GGUI viewer expects."""
    sym = str(sym or "")
    code = str(code or "")
    low_sym = sym.lower()
    if low_sym in ("space", "spacebar"):
        return SPACE
    if low_sym in ("escape", "esc"):
        return ESCAPE
    if low_sym in ("up", "uparrow"):
        return "up"
    if low_sym in ("down", "downarrow"):
        return "down"
    if low_sym in ("left", "leftarrow"):
        return "left"
    if low_sym in ("right", "rightarrow"):
        return "right"
    if low_sym in ("plus", "kp_add"):
        return "+"
    if low_sym in ("minus", "kp_subtract", "underscore"):
        return "-"
    if low_sym in ("equal", "kp_equal"):
        return "="
    if len(code) == 1 and code.isprintable() and not code.isspace():
        return code.lower() if code.isalpha() else code
    if len(low_sym) == 1:
        return low_sym
    return low_sym or code.lower()


@dataclass
class _Event:
    key: Any


class Camera:
    def __init__(self) -> None:
        self.curr_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.curr_lookat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.curr_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def position(self, x: float, y: float, z: float) -> None:
        self.curr_position = np.array([x, y, z], dtype=np.float64)

    def lookat(self, x: float, y: float, z: float) -> None:
        self.curr_lookat = np.array([x, y, z], dtype=np.float64)

    def up(self, x: float, y: float, z: float) -> None:
        self.curr_up = np.array([x, y, z], dtype=np.float64)


class _Scene:
    def __init__(self, window: "Window") -> None:
        self._window = window

    def set_camera(self, camera: Camera) -> None:
        self._window._camera = camera

    def ambient_light(self, color: tuple[float, float, float]) -> None:
        self._window._ambient = color

    def point_light(self, pos: tuple[float, float, float], color: tuple[float, float, float]) -> None:
        self._window._light_pos = pos
        self._window._light_color = color

    def particles(self, pos_field, *, radius: float, per_vertex_color, index_count: int) -> None:
        self._window._particles.append(
            (pos_field, per_vertex_color, int(index_count), float(radius))
        )

    def lines(self, vert_field, *, per_vertex_color, width: float, vertex_count: int) -> None:
        self._window._lines.append((vert_field, per_vertex_color, int(vertex_count), float(width)))


class _GUI:
    def __init__(self, window: "Window") -> None:
        self._window = window
        self._lines: list[str] = []

    def begin(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def text(self, msg: str) -> None:
        self._lines.append(str(msg))

    def end(self) -> None:
        return None


class Window:
    def __init__(self, title: str, res: tuple[int, int], vsync: bool = True) -> None:
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "Interactive viewer needs PyVista (Quadrants has no GGUI). "
                "Install with: pip install -e '.[export]'"
            ) from exc

        self.running = True
        self.GUI = _GUI(self)
        self._pv = pv
        self._plotter = pv.Plotter(window_size=list(res), title=title)
        self._plotter.set_background("black")
        self._shown = False
        self._input_bound = False
        self._events: list[_Event] = []
        self._pressed: set[str] = set()
        self._lmb = False
        self._cursor = (0.5, 0.5)
        self._camera = Camera()
        self._particles: list[tuple[Any, Any, int, float]] = []
        self._lines: list[tuple[Any, Any, int, float]] = []
        self._ambient = (0.4, 0.4, 0.4)
        self._light_pos = (0.0, 0.0, 1.0)
        self._light_color = (1.0, 1.0, 1.0)
        self._lights_key: tuple[Any, ...] | None = None
        self._hud_actor: Any = None
        self._style: Any = None

    def _vtk_iren(self) -> Any:
        iren = getattr(self._plotter, "iren", None)
        if iren is None:
            return None
        return getattr(iren, "interactor", iren)

    def _bind_input(self) -> None:
        if self._input_bound:
            return
        wrap = getattr(self._plotter, "iren", None)
        vtk_iren = self._vtk_iren()
        if wrap is None or vtk_iren is None:
            return
        try:
            if not wrap.initialized:
                wrap.initialize()
        except Exception:
            pass
        try:
            vtk_iren.Enable()
        except Exception:
            pass
        # Trackball + CharEvent steal WASD / R / F / P / Q from the GGUI contract.
        # vtkInteractorStyleUser.OnChar() falls through to the base style (q/e =
        # quit) unless CharEvent is observed on the *style* or OnChar is a no-op.
        try:
            style = _waam_interactor_style()
            vtk_iren.SetInteractorStyle(style)
            wrap._style_class = style
            self._style = style
        except Exception:
            self._style = None
        try:
            wrap.clear_key_event_callbacks()
            for key in _PYVISTA_KEYS_TO_CLEAR:
                try:
                    self._plotter.clear_events_for_key(key)
                except Exception:
                    pass
        except Exception:
            pass

        def _abort_char(*_args: Any) -> None:
            try:
                vtk_iren.SetAbortFlag(1)
            except Exception:
                pass

        def _observe(obj: Any, event: str, cb: Any, priority: float = 0.0) -> None:
            if obj is None:
                return
            try:
                obj.AddObserver(event, cb, priority)
            except Exception:
                pass

        # Bind on the VTK interactor, not PyVista's wrapper. Release events are
        # also registered on the style — vtkInteractorStyleUser has AddObserver
        # only (PyVista's add_observer assumes a wrapped style).
        _observe(vtk_iren, "KeyPressEvent", self._on_key_press)
        _observe(vtk_iren, "KeyReleaseEvent", self._on_key_release)
        _observe(vtk_iren, "LeftButtonPressEvent", self._on_lmb_press)
        _observe(vtk_iren, "LeftButtonReleaseEvent", self._on_lmb_release)
        _observe(vtk_iren, "MouseMoveEvent", self._on_mouse_move)
        _observe(vtk_iren, "ExitEvent", self._on_vtk_exit)
        _observe(vtk_iren, "CharEvent", _abort_char, 1.0)
        style = vtk_iren.GetInteractorStyle()
        _observe(style, "CharEvent", _abort_char, 1.0)
        _observe(style, "LeftButtonPressEvent", self._on_lmb_press)
        _observe(style, "LeftButtonReleaseEvent", self._on_lmb_release)
        _observe(style, "MouseMoveEvent", self._on_mouse_move)
        self._input_bound = True

    def _on_vtk_exit(self, *_args: Any) -> None:
        """VTK q/e is quit; those keys are camera down/up here. Only honor a real close."""
        key = self._key_from_vtk()
        if key in ("q", "e"):
            return
        try:
            if self._plotter.render_window is None:
                self.running = False
                return
        except Exception:
            self.running = False
            return
        # Window-manager close still fires ExitEvent with no q/e key.
        if not key:
            self.running = False

    def _key_from_vtk(self) -> str:
        vtk_iren = self._vtk_iren()
        sym = ""
        ch = ""
        if vtk_iren is not None:
            try:
                sym = str(vtk_iren.GetKeySym() or "")
            except Exception:
                sym = ""
            try:
                ch = str(vtk_iren.GetKeyCode() or "")
            except Exception:
                ch = ""
        return normalize_vtk_key(sym, ch)

    def _on_key_press(self, *_args: Any) -> None:
        key = self._key_from_vtk()
        if not key:
            return
        already = key in self._pressed
        self._pressed.add(key)
        if key == ESCAPE:
            self.running = False
        if not already:
            self._events.append(_Event(key=key))

    def _on_key_release(self, *_args: Any) -> None:
        key = self._key_from_vtk()
        self._pressed.discard(key)
        if len(key) == 1:
            self._pressed.discard(key.lower())
            self._pressed.discard(key.upper())

    def _on_lmb_press(self, *_args: Any) -> None:
        self._lmb = True
        self._pressed.add(LMB)

    def _on_lmb_release(self, *_args: Any) -> None:
        self._lmb = False
        self._pressed.discard(LMB)

    def _on_mouse_move(self, *_args: Any) -> None:
        vtk_iren = self._vtk_iren()
        if vtk_iren is None:
            return
        try:
            x, y = vtk_iren.GetEventPosition()
            w, h = self._plotter.window_size
            if w and h:
                self._cursor = (float(x) / float(w), float(y) / float(h))
        except Exception:
            pass

    def get_events(self, _kind: Any = None) -> list[_Event]:
        ev = self._events
        self._events = []
        return ev

    def is_pressed(self, key: Any) -> bool:
        if key == LMB:
            return self._lmb
        return str(key).lower() in self._pressed or str(key) in self._pressed

    def get_cursor_pos(self) -> tuple[float, float]:
        return self._cursor

    def get_canvas(self) -> "Window":
        return self

    def get_scene(self) -> _Scene:
        self._particles = []
        self._lines = []
        return _Scene(self)

    def scene(self, _scene: _Scene) -> None:
        """GGUI: commit the 3D scene onto the canvas. Geometry is already stored."""
        return None

    def save_image(self, path: str) -> None:
        self._apply_draw(consume_hud=False)
        self._plotter.screenshot(path)

    def _field_xyz(self, field: Any, n: int) -> np.ndarray:
        arr = np.asarray(field.to_numpy(), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 3))
        n = max(0, min(int(n), arr.shape[0]))
        return np.ascontiguousarray(arr[:n])

    def _point_size_px(self, radius_mm: float) -> float:
        """Approximate GGUI world-space sphere diameter in pixels."""
        cam = self._camera
        dist = float(np.linalg.norm(cam.curr_position - cam.curr_lookat))
        height = float((self._plotter.window_size or [1280, 720])[1] or 720)
        # GGUI radius is mm; fill ~one cell at the lookat distance.
        px = (2.0 * max(radius_mm, 1e-6) / max(dist, 1.0)) * height * 2.4
        return float(np.clip(px, 5.0, 28.0))

    def _sync_lights(self) -> None:
        look = tuple(float(x) for x in self._camera.curr_lookat)
        key = (tuple(self._light_pos), tuple(self._light_color), tuple(self._ambient), look)
        if key == self._lights_key:
            return
        pl = self._plotter
        try:
            pl.remove_all_lights()
        except Exception:
            pass
        try:
            lamp = self._pv.Light(
                position=self._light_pos,
                focal_point=look,
                color=self._light_color,
                intensity=1.2,
                positional=True,
            )
            pl.add_light(lamp)
            head = self._pv.Light(
                light_type="headlight",
                intensity=0.55,
                color=self._ambient,
            )
            pl.add_light(head)
            self._lights_key = key
        except Exception:
            try:
                pl.enable_lightkit()
                self._lights_key = key
            except Exception:
                pass

    def _set_hud(self, hud: str) -> None:
        """Update the overlay in place — destroying it every frame is the flicker."""
        pl = self._plotter
        actor = self._hud_actor
        if actor is None:
            actor = pl.actors.get("hud") if hasattr(pl, "actors") else None
        if hud:
            if actor is not None and hasattr(actor, "set_text"):
                actor.set_text("upper_left", hud)
                self._hud_actor = actor
                return
            self._hud_actor = pl.add_text(
                hud,
                font_size=9,
                color="white",
                font="courier",
                name="hud",
                shadow=True,
                render=False,
            )
            return
        if actor is not None:
            pl.remove_actor("hud", reset_camera=False, render=False)
            self._hud_actor = None

    def _apply_draw(self, *, consume_hud: bool = True) -> None:
        pl = self._plotter
        stale = [
            name
            for name in list(pl.actors)
            if str(name).startswith("particles") or name == "lines"
        ]
        for name in stale:
            pl.remove_actor(name, reset_camera=False, render=False)
        self._sync_lights()
        for i, (pos, col, n, radius) in enumerate(self._particles):
            if n <= 0:
                continue
            pts = self._field_xyz(pos, n)
            rgb = np.clip(self._field_xyz(col, n), 0.0, 1.0)
            if pts.shape[0] == 0:
                continue
            # Vertex cells only — add_mesh on bare points Delaunay-triangulates
            # into the dark wireframe plate (not GGUI spheres).
            nv = pts.shape[0]
            verts = np.empty(nv * 2, dtype=np.int64)
            verts[0::2] = 1
            verts[1::2] = np.arange(nv, dtype=np.int64)
            cloud = self._pv.PolyData(pts, verts=verts)
            cloud["RGB"] = rgb
            px = self._point_size_px(radius)
            pl.add_points(
                cloud,
                scalars="RGB",
                rgb=True,
                render_points_as_spheres=True,
                point_size=px,
                lighting=True,
                ambient=0.4,
                name=f"particles_{i}",
                reset_camera=False,
                render=False,
            )
        if self._lines:
            chunks: list[np.ndarray] = []
            colors: list[np.ndarray] = []
            width = 2.0
            for vert, col, nvert, w in self._lines:
                if nvert < 2:
                    continue
                pts = self._field_xyz(vert, nvert)
                rgb = self._field_xyz(col, nvert)
                chunks.append(pts)
                colors.append(np.clip(rgb, 0.0, 1.0))
                width = w
            if chunks:
                pts = np.vstack(chunks)
                rgb = np.vstack(colors)
                nseg = pts.shape[0] // 2
                lines = np.hstack(
                    [np.full((nseg, 1), 2, dtype=np.int64),
                     np.arange(nseg * 2, dtype=np.int64).reshape(nseg, 2)]
                )
                pdata = self._pv.PolyData(pts, lines=lines)
                pdata["RGB"] = rgb
                pl.add_mesh(
                    pdata,
                    scalars="RGB",
                    rgb=True,
                    line_width=max(1.0, width),
                    name="lines",
                    reset_camera=False,
                    render=False,
                )
        cam = self._camera
        pl.camera.position = tuple(float(x) for x in cam.curr_position)
        pl.camera.focal_point = tuple(float(x) for x in cam.curr_lookat)
        pl.camera.up = tuple(float(x) for x in cam.curr_up)
        hud = "\n".join(self.GUI._lines)
        if consume_hud:
            self.GUI._lines = []
        self._set_hud(hud)

    def show(self) -> None:
        self._apply_draw()
        wrap = getattr(self._plotter, "iren", None)
        if not self._shown:
            self._plotter.show(interactive_update=True, auto_close=False)
            self._shown = True
            # show() installs trackball + default q/b/v keys; bind after that.
            self._bind_input()
        else:
            self._plotter.update()
        try:
            if self._plotter.render_window is None:
                self.running = False
                return
        except Exception:
            pass
        try:
            if wrap is None:
                wrap = getattr(self._plotter, "iren", None)
            if wrap is not None:
                if not wrap.initialized:
                    wrap.initialize()
                wrap.process_events()
        except Exception:
            pass
