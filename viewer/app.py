"""Interactive melt-pool viewer (PyVista; Quadrants has no GGUI)."""

from __future__ import annotations

import pathlib
import time

import numpy as np

from waam_twin.compiler import ti

from . import pv_ui as ui
from .cli import build_parser
from .cut import CutPlane
from .probe_ui import ProbeUI
from .session import create_session
from .streamlines import seeds_in_liquid_near_torch, trace_streamlines

MODE_TEMP = 0
MODE_HAZ = 1
MODE_VEL = 2
MODE_VORT = 3
MODE_FORCE = 4
MODE_DEPOSIT = 5
MODE_NAMES = ("Temperature", "HAZ (T_max)", "Velocity", "Vorticity", "Body force", "Deposit")

SPF_MAX = 5000
TELEMETRY_EVERY_S = 1.0

FLOW_OFF = 0
FLOW_ARROWS = 1
FLOW_STREAMLINES = 2
FLOW_NAMES = ("off", "arrows", "streamlines")



class _ParticleBuffers:
    """Render buffers sized to what is drawn, not to the whole grid.

    Extract kernels count every passing cell but write only below capacity;
    when the count overflows, the buffers grow and the frame re-extracts.
    A full-grid buffer would be ~0.5 GB at dx 0.2 mm, undoing the sparse grid.
    """

    N_MARKERS = 8  # torch, arc attach, pick

    def __init__(self, capacity: int, limit: int) -> None:
        self.limit = int(limit)
        self._tree = None
        self._alloc(min(int(capacity), self.limit))

    def _alloc(self, cap: int) -> None:
        # The old tree is kept, not destroyed: SNodeTree.destroy() mid-run
        # breaks Quadrants 1.3.3's launch-context cache (assert
        # "kernel_ == other.kernel_" in the next physics kernel). Growth is
        # geometric, so the leak stays below the final size.
        fb = ti.FieldsBuilder()
        self.pos = ti.Vector.field(3, dtype=ti.f32)
        self.col = ti.Vector.field(3, dtype=ti.f32)
        fb.dense(ti.i, cap + self.N_MARKERS).place(self.pos, self.col)
        self._tree = fb.finalize()
        self.cap = cap

    def fits(self, n: int) -> bool:
        """False (after growing) when n cells did not fit: extract again."""
        if n <= self.cap or self.cap >= self.limit:
            return True
        self._alloc(min(self.limit, int(n * 1.3) + 4096))
        return False

    @property
    def max_out(self) -> int:
        return self.cap + self.N_MARKERS


def _track_camera_zup(
    camera: ui.Camera,
    window: ui.Window,
    *,
    movement_speed: float = 2.0,
    yaw_speed: float = 2.0,
    pitch_speed: float = 2.0,
    hold_key=ui.LMB,
) -> None:
    """WASD/QE + LMB look for Z-up scenes.

    Quadrants dropped GGUI; this is the same FPS contract as the old Taichi
    helper (lookat = position + front, yaw about Z, pitch elevation).
    """
    import time
    from math import asin, atan2, cos, pi, sin

    if not hasattr(camera, "_waam_last_time"):
        camera._waam_last_time = None
        camera._waam_last_mouse = (None, None)

    pos = np.asarray(camera.curr_position, dtype=np.float64)
    look = np.asarray(camera.curr_lookat, dtype=np.float64)
    front = look - pos
    fn = float(np.linalg.norm(front))
    front = front / fn if fn > 1e-12 else np.array([0.0, 1.0, 0.0])
    up = np.asarray(camera.curr_up, dtype=np.float64)
    un = float(np.linalg.norm(up))
    up = up / un if un > 1e-12 else np.array([0.0, 0.0, 1.0])
    left = np.cross(up, front)

    now = time.perf_counter_ns()
    if camera._waam_last_time is None:
        camera._waam_last_time = now
    dt = (now - camera._waam_last_time) * 1e-9
    camera._waam_last_time = now
    dt = min(max(dt, 0.0), 0.05)
    speed = movement_speed * dt * 60.0

    delta = np.zeros(3, dtype=np.float64)
    if window.is_pressed("w"):
        delta += front * speed
    if window.is_pressed("s"):
        delta -= front * speed
    if window.is_pressed("a"):
        delta += left * speed
    if window.is_pressed("d"):
        delta -= left * speed
    if window.is_pressed("e") or window.is_pressed("up"):
        delta += up * speed
    if window.is_pressed("q") or window.is_pressed("down"):
        delta -= up * speed
    if float(np.linalg.norm(delta)) > 0.0:
        camera.position(*(pos + delta))
        pos = np.asarray(camera.curr_position, dtype=np.float64)

    mx, my = window.get_cursor_pos()
    lx, ly = camera._waam_last_mouse
    holding = hold_key is None or window.is_pressed(hold_key)
    if holding:
        if lx is None or ly is None:
            pass
        else:
            dx = mx - lx
            dy = my - ly
            fz = float(np.clip(float(front[2]), -0.999, 0.999))
            pitch = asin(fz)
            yaw = atan2(float(front[0]), float(front[1]))
            yaw += dx * yaw_speed * dt * 60.0
            pitch += dy * pitch_speed * dt * 60.0
            pitch = float(np.clip(pitch, -pi / 2 * 0.99, pi / 2 * 0.99))
            front = np.array([
                sin(yaw) * cos(pitch),
                cos(yaw) * cos(pitch),
                sin(pitch),
            ], dtype=np.float64)
    camera.lookat(*(pos + front))
    camera._waam_last_mouse = (mx, my) if holding else (None, None)


def _follow_point(session, plate_top_mm: float) -> tuple[float, float, float]:
    """Camera target: torch (x, y) and the metal surface under it (z)."""
    x, y, z = session.torch_surface_mm()
    return x, y, max(z, plate_top_mm) if np.isfinite(z) else plate_top_mm


def _ensure_output_dir(path: str | None) -> pathlib.Path:
    from ..paths import PROJECT_ROOT, resolve_output_path

    if path is None:
        out = PROJECT_ROOT / "viewer_output"
    else:
        out = resolve_output_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sparkline(values, t_min: float, t_max: float, width: int = 24) -> str:
    if not values:
        return "(no data yet)"
    blocks = " ▁▂▃▄▅▆▇█"
    # NaN after a physics blow-up used to crash the viewer here.
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return "(T invalid / NaN)"
    span = max(float(t_max) - float(t_min), 1.0)
    if len(values) >= width:
        sample = list(values)[-width:]
    else:
        sample = list(values)
    chars = []
    for v in sample:
        if not np.isfinite(v):
            chars.append("?")
            continue
        idx = int(max(0, min(len(blocks) - 1, round((v - t_min) / span * (len(blocks) - 1)))))
        chars.append(blocks[idx])
    return "".join(chars)


def _flow_filter(filter_mode: int, flow_mode: int, liquid_only: int) -> int:
    """Prefer liquid/surface cells for flow overlays."""
    from .extract import FILTER_ALL, FILTER_LIQUID, FILTER_SOLID, FILTER_SURFACE

    if flow_mode == FLOW_OFF:
        return filter_mode
    if filter_mode == FILTER_SOLID:
        return FILTER_LIQUID
    if filter_mode == FILTER_ALL and liquid_only:
        return FILTER_LIQUID
    return filter_mode



def run(argv: list[str] | None = None) -> None:
    from waam_twin.runtime import init_taichi

    args = build_parser().parse_args(argv)
    init_taichi()

    from waam_twin import kernels
    from .extract import (
        FILTER_ALL,
        FILTER_LIQUID,
        FILTER_SOLID,
        FILTER_SURFACE,
        extract_body_force,
        extract_deposit,
        extract_section,
        extract_flow_arrows,
        extract_flow_arrows_near,
        extract_force_arrows,
        extract_force_arrows_near,
        extract_haz,
        extract_melt_pool,
        extract_torch_marker,
        extract_arc_attach_marker,
        extract_tracers,
        extract_velocity,
        extract_vorticity,
        reset_count,
    )

    out_dir = _ensure_output_dir(args.output_dir)

    print("=" * 60)
    print("  WAAM Digital Twin v2 — Interactive Viewer")
    print("=" * 60)

    session = create_session(
        job=args.job,
        preset=args.preset,
        material=args.material,
        auto_grid=args.auto_grid,
        sparse=args.sparse,
    )
    twin = session.twin
    g = twin.grid
    particle_scale = max(0.05, min(1.0, args.particle_scale))
    tracer_scale = particle_scale * 1.35

    count_field = ti.field(dtype=ti.i32, shape=())
    dx_mm = g.dx * 1000.0
    bufs = _ParticleBuffers(65536, g.nx * g.ny * g.nz)

    max_arrows = 4096
    arrow_vert = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)
    arrow_col = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)

    tracer_pos = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)
    tracer_col = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)

    sl_vert = ti.Vector.field(3, dtype=ti.f32, shape=32768)
    sl_col = ti.Vector.field(3, dtype=ti.f32, shape=32768)

    sec_buf = ti.field(dtype=ti.f32, shape=(max(g.nx, g.ny), g.nz))

    window = ui.Window("WAAM Digital Twin", (1280, 720), vsync=True)
    camera = ui.Camera()

    # Requested domain (a sparse grid pads axes to whole 8-cell blocks).
    nx_d = int(getattr(g, "nx_req", g.nx))
    ny_d = int(getattr(g, "ny_req", g.ny))
    nz_d = int(getattr(g, "nz_req", g.nz))
    cx = nx_d * dx_mm / 2.0 + session.offset_x_mm()
    cy = ny_d * dx_mm / 2.0
    cz = nz_d * dx_mm / 2.0
    # Aim at the plate top (arc attachment), not mid-domain Z — otherwise a
    # deep melt pool on a Clip-Y cut looks "centred" in the plate thickness.
    plate_top_mm = float(twin.nz_solid) * dx_mm
    # ~35 mm from the torch, looking down-and-along the plate: a 4 mm bead
    # fills a useful part of the window (the old domain-wide view was ~75 mm).
    cam_offset = np.array([-0.35, -0.8, 0.5]) / np.linalg.norm([-0.35, -0.8, 0.5]) * 35.0
    follow = True
    follow_target = np.array(_follow_point(session, plate_top_mm), dtype=np.float64)
    camera.position(*(follow_target + cam_offset))
    camera.lookat(*follow_target)
    camera.up(0.0, 0.0, 1.0)

    paused = args.paused
    render_mode = MODE_DEPOSIT
    flow_mode = FLOW_OFF
    auto_spf = args.steps_per_frame <= 0
    frame_target_s = max(0.02, float(args.frame_time))
    steps_per_frame = 10 if auto_spf else min(SPF_MAX, args.steps_per_frame)
    sim_rate = 0.0  # simulated ms per wall second (smoothed)
    prev_frame_t0: float | None = None
    prev_n_run = 0
    show_tracers = False  # porosity tracers sit on the pool and hide the bead (T shows them)
    show_torch = True
    show_surface_mesh = bool(twin.enable_vof) and not args.particles
    mesh_step = -1
    surface_mesh = None
    surface_key = None
    frames_since_surface = 0
    surface_announced = False
    cut = CutPlane()
    plane_edit = False  # Tab: mouse plane widget owns the left button
    filter_mode = FILTER_LIQUID if args.liquid_only else FILTER_ALL
    use_phi = 1 if twin.enable_vof else 0
    dump_idx = 0
    pending_screenshot = False
    streamline_cache: list[np.ndarray] = []
    streamline_frame = -999
    num_flow_arrows = 0
    telem: dict | None = None
    telem_wall = -1e9
    telem_step = -1
    telem_cost = 0.0
    bead: tuple[float, float, float] | None = None
    bead_station = ""

    if twin.probe_recorder is None and getattr(twin, "_job_config", None):
        probes_cfg = twin._job_config.get("probes")
        if probes_cfg:
            from waam_twin.export.probes import ProbeRecorder
            twin.probe_recorder = ProbeRecorder.from_job_list(probes_cfg, twin)
    probes = ProbeUI(twin, out_dir)
    cut_cells = (0, 0.0, 0.0, 0.0, 0.0)

    def _travel_dir() -> tuple[float, float, float]:
        d = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
        return float(d[0]), float(d[1]), float(d[2])

    def _section_lag_mm() -> float:
        return max(3.0, 0.6 * float((telem or {}).get("pool_length_mm", 0.0) or 0.0))

    def _place_probe(px_py) -> None:
        o, d = window.pixel_ray(*px_py)
        hit = probes.pick(o, d, cut_cells)
        if hit is None:
            print("[viewer] Probe: no metal under that point")
            return
        name = probes.add(hit)
        print(f"[viewer] Probe {name} at ({hit[0]:.1f}, {hit[1]:.1f}, {hit[2]:.1f}) mm"
              f"  ({'rides torch' if probes.follow_torch else 'fixed'})  → {out_dir / 'probes.yaml'}")

    _print_controls()

    print(f"\n  Job     : {session.job_label}")
    print(f"  Hardware: {getattr(twin, 'preset_name', '?')}  "
          f"(dx is the job request; device OOMs if the mesh does not fit)")
    print(f"  Domain  : {g.nx * dx_mm:.1f}×{g.ny * dx_mm:.1f}×{g.nz * dx_mm:.1f} mm  "
          f"grid {g.nx}×{g.ny}×{g.nz}  dx={dx_mm:.3f} mm")
    ps = getattr(twin, "plate_size_mm", None)
    if ps:
        print(f"  Plate   : {ps[0]:.0f}×{ps[1]:.0f}×{float(twin.plate_thickness_mm or 0):.1f} mm  "
              f"(square from top if L=W; side view shows L×thickness)")
    else:
        print(f"  Plate   : full domain XY × {float(twin.nz_solid) * dx_mm:.1f} mm thick")
    print(f"  Physics : tier={getattr(twin, 'physics_tier', '?')}  VOF={twin.enable_vof}")
    print(f"  Path    : {session.uses_path}  travel={session.torch_spd_m_s * 1000:.2f} mm/s")
    print(f"  Particles: radius={particle_scale:.2f}×dx  (--particle-scale)")
    print(f"  Output  : {out_dir.resolve()}")
    print(f"  Probes  : {(out_dir / 'probes.csv').resolve()}  (keeps recording after arc off)\n")

    def _flush_probes() -> None:
        rec = getattr(twin, "probe_recorder", None)
        if rec is None or not rec.probes:
            return
        try:
            rec.flush_csv(out_dir / "probes.csv")
        except Exception as exc:
            print(f"[viewer] Probe log write failed: {exc}")

    while window.running:
        for e in window.get_events(ui.PRESS):
            if e.key == ui.SPACE:
                paused = not paused
            elif e.key in ("m", "M"):
                render_mode = (render_mode + 1) % len(MODE_NAMES)
            elif e.key in ("v", "V"):
                flow_mode = (flow_mode + 1) % len(FLOW_NAMES)
                print(f"[viewer] Flow overlay → {FLOW_NAMES[flow_mode]}")
            elif e.key in ("b", "B"):
                if filter_mode == FILTER_ALL:
                    filter_mode = FILTER_LIQUID
                elif filter_mode == FILTER_LIQUID:
                    filter_mode = FILTER_ALL
                else:
                    filter_mode = FILTER_LIQUID
            elif e.key in ("f", "F"):
                if twin.enable_vof:
                    if filter_mode == FILTER_SURFACE:
                        filter_mode = FILTER_LIQUID
                    else:
                        filter_mode = FILTER_SURFACE
            elif e.key in ("h", "H"):
                filter_mode = FILTER_SOLID if filter_mode != FILTER_SOLID else FILTER_LIQUID
            elif e.key in ("n", "N"):
                if not twin.enable_vof:
                    show_surface_mesh = False
                    print("[viewer] φ mesh needs enable_vof")
                else:
                    show_surface_mesh = not show_surface_mesh
                    surface_mesh = None
                    surface_key = None
                    frames_since_surface = 5
                    surface_announced = False
                    print(
                        "[viewer] Surface view → "
                        + ("φ = 0.5 mesh" if show_surface_mesh else "particles")
                    )
            elif e.key in ("x", "X", "y", "Y", "z", "Z"):
                if plane_edit:
                    plane_edit = False
                    window.set_plane_widget(False)
                cut.select("xyz".index(e.key.lower()), _travel_dir(), _section_lag_mm(),
                           z_offset_mm=-0.5 * dx_mm)
            elif e.key in ("[", "]", "{", "}"):
                step = dx_mm if e.key in ("[", "]") else 1.0
                cut.nudge(step if e.key in ("]", "}") else -step)
            elif e.key in ("k", "K"):
                cut.toggle_anchor(session.torch_surface_mm(), plate_top_mm)
            elif e.key == "\\":
                cut.toggle_flip()
            elif e.key == "tab":
                plane_edit = not plane_edit
                if plane_edit:
                    tmm = np.array(session.torch_surface_mm())
                    pl_now = cut.plane(tmm, plate_top_mm, camera.curr_position)
                    n0, o0 = pl_now if pl_now is not None else (np.array([1.0, 0.0, 0.0]), tmm)
                    cut.set_free(n0, o0)
                    ox = session.offset_x_mm()
                    window.set_plane_widget(
                        True, normal=n0, origin=o0,
                        bounds=(ox, ox + nx_d * dx_mm, 0.0, ny_d * dx_mm, 0.0, nz_d * dx_mm),
                        callback=lambda n, o: cut.set_free(n, o),
                    )
                    print("[viewer] Plane edit: drag the plane / its arrow with the left button; Tab to finish")
                else:
                    window.set_plane_widget(False)
            elif e.key in ("t", "T"):
                show_tracers = not show_tracers
            elif e.key in ("o", "O"):
                show_torch = not show_torch
            elif e.key in ("r", "R"):
                session.reset_motion()
                streamline_cache = []
                if twin.probe_recorder is not None:
                    twin.probe_recorder.clear()
                surface_mesh = None
                surface_key = None
                frames_since_surface = 5
                surface_announced = False
            elif e.key == "=" or e.key == "+":
                auto_spf = False
                steps_per_frame = min(SPF_MAX, max(steps_per_frame + 1, int(steps_per_frame * 1.5)))
            elif e.key == "-" or e.key == "_":
                auto_spf = False
                steps_per_frame = max(1, int(steps_per_frame / 1.5))
            elif e.key == "0":
                auto_spf = True
                print(f"[viewer] Steps per frame → auto ({frame_target_s:.2f} s of physics per frame)")
            elif e.key in ("c", "C"):
                follow = not follow
                print(f"[viewer] Camera follow → {'torch' if follow else 'off'}")
            elif e.key in ("p", "P"):
                # Defer to end-of-frame: save_image after the scene is drawn
                # but before window.show().
                pending_screenshot = True
            elif e.key in ("g", "G"):
                _flush_probes()
                try:
                    session.export_g_bundle(out_dir)
                except Exception as exc:
                    print(f"[viewer] Bundle export failed: {exc}")
                rec = getattr(twin, "probe_recorder", None)
                if rec is not None and rec._rows:
                    last = rec._rows[-1]
                    print(
                        f"[viewer] Probe log {len(rec._rows)} rows  "
                        f"last t={last.get('sim_time_ms')} ms  arc_on={last.get('arc_on')}  "
                        f"-> {(out_dir / 'probes.csv').resolve()}"
                    )
            elif e.key in ("u", "U"):
                name = probes.add_under_torch(session.torch_surface_mm())
                print(f"[viewer] Probe {name} {probes.rec.probes[-1].z_m * 1000:.1f} mm high, "
                      f"under the torch ({'rides torch' if probes.follow_torch else 'fixed'})")
            elif e.key in ("i", "I"):
                _place_probe(window.centre_px())
            elif e.key == ui.RMB:
                if window.last_click_px is not None:
                    _place_probe(window.last_click_px)
            elif e.key in ("l", "L"):
                probes.follow_torch = not probes.follow_torch
                print(f"[viewer] New probes → {'ride with the torch' if probes.follow_torch else 'fixed in the world'}")
            elif e.key in ("j", "J"):
                probes.cycle()
            elif e.key in ("delete", "backspace"):
                gone = probes.remove_selected()
                if gone:
                    print(f"[viewer] Probe {gone} removed")
            elif e.key == ui.ESCAPE:
                window.running = False

        frame_t0 = time.perf_counter()
        n_run = 0
        if not paused:
            n_run = steps_per_frame
            session.advance_physics(n_run, is_welding=True)
            ti.sync()
            phys_s = time.perf_counter() - frame_t0
            if auto_spf:
                # Toward frame_target_s of physics; at most 2x per frame so a
                # JIT compile or a hiccup does not swing it.
                ratio = frame_target_s / max(phys_s, 1e-3)
                steps_per_frame = int(np.clip(round(n_run * min(2.0, max(0.5, ratio))), 1, SPF_MAX))
            session.maybe_auto_export_on_path_end(out_dir)
        _flush_probes()

        # Export/diagnostic buffers are allocated lazily (not on minimal grids).
        g.ensure_export_buffers()
        if render_mode == MODE_VORT:  # 0.7 ms, but only this view reads it
            kernels.compute_vorticity_magnitude(
                g.ux, g.uy, g.uz, g.flags, g.vorticity_mag,
                g.dx, g.dt, g.FLAG_GAS, g.nx, g.ny, g.nz,
            )

        # Camera follows the torch (x, y) and the bead top under it (z, smoothed);
        # the user's orbit offset is kept.
        tgt = np.array(_follow_point(session, plate_top_mm), dtype=np.float64)
        tgt[2] = follow_target[2] + 0.15 * (tgt[2] - follow_target[2])
        if follow:
            d = tgt - follow_target
            if float(np.abs(d).max()) > 1e-6:
                camera.position(*(np.asarray(camera.curr_position, dtype=np.float64) + d))
                camera.lookat(*(np.asarray(camera.curr_lookat, dtype=np.float64) + d))
        follow_target = tgt
        canvas = window.get_canvas()
        scene = window.get_scene()
        # Z-up orbit (stock trackball is Y-up and swaps axes here).
        _track_camera_zup(
            camera,
            window,
            movement_speed=2.0,
            hold_key="__plane_edit__" if plane_edit else ui.LMB,
            yaw_speed=2.0,
            pitch_speed=2.0,
        )
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(cx, cy - 30.0, cz + 40.0), color=(1.0, 0.95, 0.85))

        offset_x_mm = session.offset_x_mm()
        # Section plane: normal into the kept side, origin (world mm).
        torch_mm = np.array(session.torch_surface_mm(), dtype=np.float64)
        cut_plane = cut.plane(torch_mm, plate_top_mm, camera.curr_position)
        if cut_plane is not None:
            win_off_mm = np.array(twin.window_offset_m(), dtype=np.float64) * 1000.0
            cut_cells = (1, *cut.cell_plane(cut_plane[0], cut_plane[1], win_off_mm, dx_mm))
            # Mesh key: the plane to half a cell (a torch-riding plane moves a
            # little every frame; rebuild when that is visible).
            cut_key = (tuple(np.round(cut_plane[0], 3)), tuple(np.round(cut_plane[1] / (0.5 * dx_mm))))
        else:
            cut_cells = (0, 0.0, 0.0, 0.0, 0.0)
            cut_key = None

        if show_surface_mesh and twin.enable_vof:
            if not paused:
                frames_since_surface += 1
            deposit_mesh = render_mode == MODE_DEPOSIT
            key = (int(twin.nz_solid), deposit_mesh, cut_key)
            due = (
                surface_mesh is None
                or key != surface_key
                or (not paused and (frames_since_surface >= 5 or twin._step_n - mesh_step >= 100))
            )
            if due:
                frames_since_surface = 0
                try:
                    from .surface import phi_isosurface
                    built = phi_isosurface(
                        g.phi.to_numpy(),
                        g.T.to_numpy(),
                        dx_mm=dx_mm,
                        offset_x_mm=offset_x_mm,
                        nz_solid=int(twin.nz_solid),
                        t_lo=float(twin.T_amb),
                        t_hi=float(twin.mat.T_liquidus) + 800.0,
                        cut=cut_plane,
                        alloy_frac=g.alloy_frac.to_numpy() if deposit_mesh else None,
                        f_l=g.f_l.to_numpy() if deposit_mesh else None,
                        t_solidus=float(twin.mat.T_solidus),
                    )
                except Exception as exc:
                    built = None
                    if not surface_announced:
                        print(f"[viewer] φ mesh failed: {exc}")
                        surface_announced = True
                mesh_step = twin._step_n
                surface_key = key
                surface_mesh = built
                if built is not None and not surface_announced:
                    print(f"[viewer] φ=0.5 mesh  {built.n_cells} tris")
                    surface_announced = True
            scene.surface(surface_mesh)
        else:
            scene.surface(None)
        draw_cells = not (show_surface_mesh and twin.enable_vof)
        clip_args = cut_cells

        def _extract_cells() -> None:
            render_pos, render_col, max_cells = bufs.pos, bufs.col, bufs.cap
            if render_mode == MODE_DEPOSIT:
                extract_deposit(
                    g.alloy_frac, g.f_l, g.T, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm,
                    twin.mat.T_solidus, twin.mat.T_liquidus,
                    g.FLAG_GAS, g.FLAG_SOLID,
                    filter_mode, use_phi, max_cells, *clip_args,
                )
            elif render_mode == MODE_TEMP:
                extract_melt_pool(
                    g.f_l, g.T, g.T_max, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm,
                    float(twin.T_amb), twin.mat.T_solidus, twin.mat.T_liquidus,
                    twin.nz_solid,
                    g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                    filter_mode, use_phi, max_cells, *clip_args,
                )
            elif render_mode == MODE_HAZ:
                extract_haz(
                    g.T_max, g.f_l, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm,
                    twin.mat.T_solidus, g.FLAG_GAS, g.FLAG_SOLID,
                    filter_mode, use_phi, max_cells, *clip_args,
                )
            elif render_mode == MODE_VEL:
                extract_velocity(
                    g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm, g.dx, g.dt, g.u_ref_phys,
                    g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                    filter_mode, use_phi, max_cells, *clip_args,
                )
            elif render_mode == MODE_VORT:
                # Adaptive scale: fixed 5000/s made real pool vorticity (~10–300/s) invisible.
                vort_np = g.vorticity_mag.to_numpy()
                fl_np = g.f_l.to_numpy()
                mask = fl_np > 0.12
                if np.any(mask):
                    vals = vort_np[mask]
                    vort_ref = float(np.percentile(vals, 95))
                    vort_floor = float(np.percentile(vals, 40))
                else:
                    vmax = float(np.max(vort_np)) if vort_np.size else 0.0
                    vort_ref = max(vmax, 50.0)
                    vort_floor = 0.05 * vort_ref
                vort_ref = max(vort_ref, 5.0)
                vort_floor = max(min(vort_floor, 0.35 * vort_ref), 0.5)
                # Prefer liquid so cold solid does not wash out the field.
                vort_filter = FILTER_LIQUID if filter_mode == FILTER_ALL else filter_mode
                extract_vorticity(
                    g.vorticity_mag, g.f_l, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm, vort_ref, vort_floor,
                    g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                    vort_filter, use_phi, max_cells, *clip_args,
                )
            else:
                extract_body_force(
                    g.Fx_snap, g.Fy_snap, g.Fz_snap,
                    g.f_l, g.phi, g.flags,
                    render_pos, render_col, count_field,
                    dx_mm, offset_x_mm, 0.01,
                    g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                    filter_mode, use_phi, max_cells, *clip_args,
                )

        reset_count(count_field)
        if draw_cells:
            _extract_cells()
            if not bufs.fits(int(count_field[None])):
                reset_count(count_field)
                _extract_cells()
            if int(count_field[None]) > bufs.cap:
                count_field[None] = bufs.cap
        render_pos, render_col = bufs.pos, bufs.col
        num_cells = count_field[None]

        if show_torch:
            # Cyan = arc attachment / heat inject (plate surface under torch).
            # Yellow = contact tip (surface + CTWD). Heat is at cyan, not yellow.
            sx, sy, sz = session.torch_surface_mm()
            extract_arc_attach_marker(
                render_pos, render_col, count_field,
                sx, sy, sz, bufs.max_out,
            )
            tx, ty, tz = session.torch_marker_mm()
            extract_torch_marker(
                render_pos, render_col, count_field,
                tx, ty, tz, bufs.max_out,
            )
            num_cells = count_field[None]

        for pts, texts, colour in probes.markers():
            scene.labels(pts, texts, color=colour, size_px=12.0)

        if num_cells > 0:
            rad = dx_mm * particle_scale * (0.55 if show_surface_mesh else 1.0)
            scene.particles(
                render_pos,
                radius=rad,
                per_vertex_color=render_col,
                index_count=num_cells,
            )

        num_flow_arrows = 0
        if flow_mode == FLOW_ARROWS:
            tx, ty, tz = session.torch_surface_mm()
            ti_c = int(np.clip((tx - offset_x_mm) / dx_mm, 0, g.nx - 1))
            tj_c = int(np.clip(ty / dx_mm, 0, g.ny - 1))
            tk_c = int(np.clip(tz / dx_mm, 0, g.nz - 1))
            reset_count(count_field)
            extract_flow_arrows_near(
                g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
                arrow_vert, arrow_col, count_field,
                dx_mm, offset_x_mm, g.dx, g.dt,
                dx_mm * 4.0, dx_mm * 1.5,
                ti_c, tj_c, tk_c, 12, 1,
                g.FLAG_GAS, max_arrows, g.nx, g.ny, g.nz,
            )
            num_flow_arrows = count_field[None]
            if num_flow_arrows == 0:
                reset_count(count_field)
                extract_flow_arrows(
                    g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
                    arrow_vert, arrow_col, count_field,
                    dx_mm, offset_x_mm, g.dx, g.dt,
                    dx_mm * 4.0, dx_mm * 1.5,
                    1,
                    g.FLAG_GAS, g.FLAG_SOLID,
                    FILTER_LIQUID, use_phi, max_arrows,
                    0, 0.0, 0.0, 0.0, 0.0,
                )
                num_flow_arrows = count_field[None]
            if num_flow_arrows > 0:
                scene.lines(
                    arrow_vert,
                    per_vertex_color=arrow_col,
                    width=5.0,
                    vertex_count=num_flow_arrows * 2,
                )

        if render_mode == MODE_FORCE and flow_mode == FLOW_OFF:
            tx, ty, tz = session.torch_surface_mm()
            ti_c = int(np.clip((tx - offset_x_mm) / dx_mm, 0, g.nx - 1))
            tj_c = int(np.clip(ty / dx_mm, 0, g.ny - 1))
            tk_c = int(np.clip(tz / dx_mm, 0, g.nz - 1))
            reset_count(count_field)
            extract_force_arrows_near(
                g.Fx_snap, g.Fy_snap, g.Fz_snap,
                g.f_l, g.phi, g.flags,
                arrow_vert, arrow_col, count_field,
                dx_mm, offset_x_mm, dx_mm * 3.5, 0.002,
                ti_c, tj_c, tk_c, 10, 1,
                g.FLAG_GAS, max_arrows, g.nx, g.ny, g.nz,
            )
            n_force = count_field[None]
            if n_force > 0:
                scene.lines(
                    arrow_vert,
                    per_vertex_color=arrow_col,
                    width=3.5,
                    vertex_count=n_force * 2,
                )

        if flow_mode == FLOW_STREAMLINES and twin._step_n - streamline_frame > 10:
            streamline_frame = twin._step_n
            tx, ty, tz = session.torch_surface_mm()
            ti_c = int(np.clip((tx - offset_x_mm) / dx_mm, 0, g.nx - 1))
            tj_c = int(np.clip(ty / dx_mm, 0, g.ny - 1))
            tk_c = int(np.clip(tz / dx_mm, 0, g.nz - 1))
            ux_np = g.ux.to_numpy()
            uy_np = g.uy.to_numpy()
            uz_np = g.uz.to_numpy()
            fl_np = g.f_l.to_numpy()
            flags_np = g.flags.to_numpy()
            seeds = seeds_in_liquid_near_torch(
                ux_np, uy_np, uz_np, fl_np, flags_np,
                ti_c, tj_c, tk_c, g.FLAG_GAS, n=36, search_r=14,
            )
            streamline_cache = trace_streamlines(
                ux_np, uy_np, uz_np, seeds,
                n_steps=160, step_cells=0.4,
                f_l=fl_np, flags=flags_np, flag_gas=g.FLAG_GAS,
                fl_cut=0.04, bidirectional=True,
            )

        if flow_mode == FLOW_STREAMLINES and streamline_cache:
            # scene.lines reads GPU buffers — host index writes to
            # ti.Vector.field often do not upload. Build numpy then from_numpy.
            max_verts = sl_vert.shape[0]
            vert_np = np.zeros((max_verts, 3), dtype=np.float32)
            col_np = np.zeros((max_verts, 3), dtype=np.float32)
            base = 0
            n_seg = 0
            for line in streamline_cache:
                n_pts = len(line)
                if n_pts < 2:
                    continue
                for k in range(n_pts - 1):
                    if base + 1 >= max_verts:
                        break
                    p0 = line[k]
                    p1 = line[k + 1]
                    vert_np[base] = [
                        (float(p0[0]) + 0.5) * dx_mm + offset_x_mm,
                        (float(p0[1]) + 0.5) * dx_mm,
                        (float(p0[2]) + 0.5) * dx_mm,
                    ]
                    vert_np[base + 1] = [
                        (float(p1[0]) + 0.5) * dx_mm + offset_x_mm,
                        (float(p1[1]) + 0.5) * dx_mm,
                        (float(p1[2]) + 0.5) * dx_mm,
                    ]
                    # Speed-tint along the polyline (head brighter)
                    t = k / max(n_pts - 2, 1)
                    c = (0.1 + 0.2 * t, 0.75 + 0.2 * t, 1.0)
                    col_np[base] = c
                    col_np[base + 1] = c
                    base += 2
                    n_seg += 1
            if n_seg > 0:
                sl_vert.from_numpy(vert_np)
                sl_col.from_numpy(col_np)
                scene.lines(
                    sl_vert,
                    per_vertex_color=sl_col,
                    width=4.0,
                    vertex_count=n_seg * 2,
                )

        reset_count(count_field)
        extract_tracers(
            g.porosity_pos, g.porosity_active,
            tracer_pos, tracer_col, count_field,
            g.max_tracers, g.max_tracers,
            offset_x_mm, g.dx, *cut_cells,
        )
        num_tracers = count_field[None]
        if show_tracers and num_tracers > 0:
            scene.particles(
                tracer_pos,
                radius=dx_mm * tracer_scale,
                per_vertex_color=tracer_col,
                index_count=num_tracers,
            )

        canvas.scene(scene)

        # Full telemetry copies whole fields to the host (~0.1 s at dx 0.4 mm
        # without the force sampling): refresh it at most every
        # TELEMETRY_EVERY_S, and keep it under ~1/4 of the wall time.
        now_wall = time.perf_counter()
        if telem is None or (
            twin._step_n != telem_step
            and now_wall - telem_wall >= max(TELEMETRY_EVERY_S, 4.0 * telem_cost)
        ):
            telem = twin.get_telemetry(force_diagnostics=False)
            telem_cost = time.perf_counter() - now_wall
            telem_wall, telem_step = now_wall, twin._step_n
            # Behind the arc, in the trailing part of the pool where the bead
            # has its shape (liquid counts toward the height).
            lag_mm = _section_lag_mm()
            plane = session.section_plane(lag_mm)
            where = f"{lag_mm:.1f} behind arc"
            # A cut across the travel direction is where you are looking:
            # measure the bead there instead.
            tdir = _travel_dir()
            if cut.axis in (0, 1) and abs(tdir[cut.axis]) >= abs(tdir[1 - cut.axis]):
                a = cut.axis
                pos_mm = cut.position_mm(torch_mm, plate_top_mm)
                off_a = twin.window_offset_m()[a] * 1000.0
                idx_a = int((pos_mm - off_a) / dx_mm)
                n_a = g.nx if a == 0 else g.ny
                if 0 <= idx_a < n_a:
                    tc = session.torch_cell()
                    plane = (a, idx_a, tc[1 - a], pos_mm)
                    where = "at the cut"
            bead, bead_station = None, ""
            probes.refresh_summary(float(twin.mat.T_solidus))
            if probes.rec.probes:
                try:
                    probes.rec.write_summary_csv(out_dir / "probes_summary.csv", float(twin.mat.T_solidus))
                except Exception as exc:
                    print(f"[viewer] Probe summary write failed: {exc}")
            if plane is not None:
                from .bead_section import measure as _bead_measure
                axis, idx, t_c, station = plane
                n_t = g.ny if axis == 0 else g.nx
                extract_section(g.phi, g.flags, sec_buf, axis, idx, n_t, g.nz, g.FLAG_GAS)
                bead = _bead_measure(sec_buf.to_numpy()[:n_t], int(twin.nz_solid), t_c, dx_mm)
                bead_station = f"{'x' if axis == 0 else 'y'}={station:.1f} mm, {where}"
        filter_labels = {
            FILTER_ALL: "all metal",
            FILTER_LIQUID: "liquid",
            FILTER_SURFACE: "surface",
            FILTER_SOLID: "solid (HAZ)",
        }
        # Rate over the whole loop period (physics + render of the last frame).
        period_s = frame_t0 - prev_frame_t0 if prev_frame_t0 is not None else 0.0
        prev_frame_t0 = frame_t0
        if prev_n_run > 0 and period_s > 0:
            r = prev_n_run * g.dt * 1000.0 / period_s
            sim_rate = r if sim_rate <= 0 else 0.8 * sim_rate + 0.2 * r
        prev_n_run = n_run
        spf_label = f"{steps_per_frame} st/frame{' (auto)' if auto_spf else ''}"
        status_str = "PAUSED" if paused else f"LIVE  {spf_label}  {sim_rate:.2f} ms sim/s"
        if session.path_complete():
            status_str = ("PAUSED" if paused else "LIVE") + "  ARC OFF (path end)"

        window.GUI.begin("WAAM Twin", 0.01, 0.01, 0.42, 0.46)
        window.GUI.text(f"Job    : {session.job_label}")
        _ps = getattr(twin, "plate_size_mm", None)
        if _ps:
            window.GUI.text(
                f"Plate  : {_ps[0]:.0f}×{_ps[1]:.0f}×"
                f"{float(twin.plate_thickness_mm or twin.nz_solid * dx_mm):.1f} mm  "
                f"dx={dx_mm:.3f}"
            )
        else:
            window.GUI.text(f"Plate  : full XY  dx={dx_mm:.3f} mm")
        window.GUI.text(f"Status : {status_str}")
        window.GUI.text(f"View   : {MODE_NAMES[render_mode]}")
        flow_hint = FLOW_NAMES[flow_mode]
        if flow_mode != FLOW_OFF:
            flow_hint += f"  ({num_flow_arrows} arrows)" if flow_mode == FLOW_ARROWS else f"  ({len(streamline_cache)} lines)"
        window.GUI.text(f"Flow   : {flow_hint}  (V=cycle)")
        if flow_mode != FLOW_OFF and num_flow_arrows == 0:
            window.GUI.text("  (no flow arrows at torch — try B=liquid, Z=off)")
        window.GUI.text(f"Filter : {filter_labels.get(filter_mode, '?')}  (B/H/F)")
        window.GUI.text(
            f"Surface: {'φ = 0.5 mesh' if show_surface_mesh else 'particles'}  (N)  "
            f"camera {'follows torch' if follow else 'free'} (C)"
        )
        if show_surface_mesh and render_mode not in (MODE_TEMP, MODE_DEPOSIT):
            window.GUI.text("  (mesh shows T; N = particles for this view)")
        window.GUI.text(f"Cut    : {cut.label(torch_mm, plate_top_mm, _travel_dir())}")
        if cut.active or plane_edit:
            window.GUI.text(
                "  [ ] move 1 cell  { } 1 mm  K torch/fixed  \\ flip side  Tab "
                + ("finish plane edit" if plane_edit else "tilt with mouse")
            )
        if render_mode == MODE_TEMP:
            window.GUI.text("  T colors: blue→cyan→yellow→red→white")
        elif render_mode == MODE_DEPOSIT:
            window.GUI.text("  copper = wire metal, grey = plate, orange→white = liquid")
        # Arc vs plate — proves heat tracks torch at the surface (no path Z offset).
        _arc = getattr(twin, "_last_arc_ijk", None)
        if _arc is not None:
            _arc_z = float(_arc[2]) * dx_mm
            _plate_z = float(twin.nz_solid) * dx_mm
            window.GUI.text(
                f"Arc Z  : {_arc_z:.1f} mm  plate top {_plate_z:.1f} mm  "
                f"(cyan=attach yellow=tip)"
            )
        window.GUI.text(f"Sim t  : {twin._step_n * g.dt * 1000:.2f} ms  step {twin._step_n}")
        window.GUI.text(
            f"Pool   : W {telem['pool_width_mm']:.2f} mm  "
            f"D {telem['pool_depth_mm']:.2f} mm  "
            f"L {telem.get('pool_length_mm', 0):.1f} mm"
        )
        window.GUI.text(
            f"T_peak : {telem['peak_temp_C']:.0f} °C  "
            f"u_max {telem['marangoni_vel_ms']:.3f} m/s"
        )
        cool = telem.get("peak_cooling_rate_Ks", 0.0)
        window.GUI.text(f"dT/dt  : peak cool {cool:.0f} K/s  (metal)")
        if not np.isfinite(telem.get("peak_temp_C", 0.0)):
            window.GUI.text("  WARN: T_peak is NaN — physics blew up; pause / R reset")
        t_cap_c = twin.T_vapor_cap_K - 273.15
        if telem.get("vapor_cap_saturated"):
            n_cap = telem.get("n_cells_at_vapor_cap", 0)
            window.GUI.text(
                f"  WARN: vapor-cap saturated ({n_cap} cells @ {t_cap_c:.0f} °C) — "
                f"peak is clamped, not physical T"
            )
        elif np.isfinite(telem["peak_temp_C"]) and telem["peak_temp_C"] > t_cap_c + 50.0:
            window.GUI.text(
                f"  WARN: T_peak above vapor cap ({t_cap_c:.0f} °C) — check enthalpy clamp"
            )
        u_lu_cap = twin.u_mach_limit_lu * g.dx / g.dt
        if np.isfinite(telem["marangoni_vel_ms"]) and telem["marangoni_vel_ms"] > 0.95 * u_lu_cap:
            window.GUI.text(
                f"  NOTE: u at LBM Mach cap (~{u_lu_cap:.2f} m/s) — forces limited"
            )
        if flow_mode == FLOW_STREAMLINES and not streamline_cache:
            window.GUI.text("  (no streamlines — need liquid + flow; wait for melt)")
        if render_mode == MODE_VORT and num_cells == 0:
            window.GUI.text("  (no vorticity — need liquid pool; try B=liquid, clip off)")
        if render_mode == MODE_VORT and num_cells > 0:
            window.GUI.text("  Vorticity: purple→yellow = |∇×u| in liquid (auto-scaled)")
        drawn = (
            f"mesh {surface_mesh.n_cells} tris" if not draw_cells and surface_mesh is not None
            else f"{num_cells} cells drawn"
        )
        window.GUI.text(f"Metal  : {drawn}  |  liquid {telem.get('n_liquid_cells', 0)}")
        dep = telem.get("deposited_mass_g", 0.0)
        exp = telem.get("expected_wire_mass_g", 0.0)
        mbal = telem.get("mass_balance_ratio", 0.0)
        if bead is not None and bead[1] > 0:
            window.GUI.text(
                f"Bead   : h {bead[0]:.2f} mm  w {bead[1]:.2f} mm  toe≈{bead[2]:.0f}°  "
                f"({bead_station})"
            )
        else:
            window.GUI.text("Bead   : — (section waits until the arc has passed)")
        window.GUI.text(
            f"Mass   : dep {dep:.3f} g / wire {exp:.3f} g  "
            f"bal {mbal:.2f}  ovf {telem.get('deposition_overflow_count', 0)}"
        )
        window.GUI.text(
            f"Phys   : {telem.get('physics_tier', '?')}  "
            f"p_arc {telem.get('arc_pressure_peak_pa', 0):.0f} Pa  "
            f"recoil {'ON' if twin.enable_recoil else 'off'}"
        )
        window.GUI.text(
            f"Clamp  : F {telem.get('force_clamp_hits_step', 0)}  "
            f"Ma {telem.get('mach_clamp_hits_step', 0)}  "
            f"(cum F {telem.get('force_clamp_hits_cum', 0)}  "
            f"Ma {telem.get('mach_clamp_hits_cum', 0)})"
        )
        if telem.get("lorentz_unconverged_streak", 0) > 0:
            window.GUI.text(
                f"  Lorentz streak {telem['lorentz_unconverged_streak']}  "
                f"(total {telem.get('lorentz_unconverged_count', 0)})"
            )
        window.GUI.text(
            f"Mat    : {telem['material_name']} ({telem['material_status']})  "
            f"tracers {num_tracers}"
        )
        for line in probes.hud_lines():
            window.GUI.text(line)
        spark_name, spark_T = probes.spark()
        if spark_T:
            lo, hi = min(spark_T) - 50.0, max(spark_T) + 50.0
            window.GUI.text(
                f"T(t) {spark_name}: {_sparkline(spark_T, lo, hi, width=32)}  "
                f"{lo - 273.15:.0f}–{hi - 273.15:.0f} °C"
            )
        window.GUI.end()

        if pending_screenshot:
            path = out_dir / f"frame_{dump_idx:05d}.png"
            try:
                # Must run after canvas.scene / GUI, before show.
                window.save_image(str(path))
                print(f"[viewer] Screenshot → {path}")
                dump_idx += 1
            except Exception as exc:
                print(f"[viewer] Screenshot failed: {exc}")
            pending_screenshot = False

        window.show()


def _print_controls() -> None:
    print("\nControls:")
    print("  SPACE     Pause / resume")
    print("  M         Cycle colour (Deposit / T / HAZ / Velocity / Vorticity / Body force)")
    print("  V         Cycle flow overlay (off / arrows / streamlines)")
    print("  B / H / F Filter: all / solid / surface")
    print("  N         Toggle φ = 0.5 bead surface (default) / cell particles")
    print("  X / Y / Z Section plane on that axis (again = off). X/Y across the travel")
    print("            start 3 mm behind the arc; Y along the bead; Z at the plate face")
    print("  [ ] { }   Move the plane 1 cell / 1 mm      K  torch-riding / fixed in world")
    print("  \\         Keep the other side               Tab  tilt the plane with the mouse")
    print("  C         Camera follows the torch on / off")
    print("  T / O     Toggle tracers / torch marker")
    print("  G         VTK bundle snapshot. Live probe trace is probes.csv")
    print("            (that file keeps recording after the arc goes off)")
    print("  RMB click Add a probe on the metal under the cursor (honours the cut)")
    print("  I / U     Add a probe at the screen centre / 1 mm under the torch")
    print("  L         New probes fixed in the world / riding with the torch")
    print("  J / Del   Select next probe (T(t) graph) / remove it")
    print("            probes.yaml (job block), probes.csv, probes_summary.csv → output dir")
    print("  R         Reset simulation")
    print("  + / -     More / fewer steps per frame (manual);  0  back to auto")
    print("  P         Screenshot → viewer_output/")
    print("  WASD      Move  Q/E or arrows  Down / up")
    print("  LMB drag  Look (Z-up)   ESC  Exit\n")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
