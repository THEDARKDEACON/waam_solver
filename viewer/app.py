"""Taichi GGUI main loop for the WAAM melt-pool viewer."""

from __future__ import annotations

import pathlib
from collections import deque

import numpy as np
import taichi as ti

from .cli import build_parser
from .pick import lookat_to_grid, sample_cell
from .session import create_session
from .streamlines import seeds_near_torch, trace_streamlines

MODE_TEMP = 0
MODE_HAZ = 1
MODE_VEL = 2
MODE_VORT = 3
MODE_FORCE = 4
MODE_NAMES = ("Temperature", "HAZ (T_max)", "Velocity", "Vorticity", "Body force")

FLOW_OFF = 0
FLOW_ARROWS = 1
FLOW_STREAMLINES = 2
FLOW_NAMES = ("off", "arrows", "streamlines")

PROBE_HISTORY_LEN = 48


def _ensure_output_dir(path: str | None) -> pathlib.Path:
    if path is None:
        from ..paths import PROJECT_ROOT
        out = PROJECT_ROOT / "viewer_output"
    else:
        out = pathlib.Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sparkline(values: deque[float], t_min: float, t_max: float, width: int = 24) -> str:
    if not values:
        return "(no data yet)"
    blocks = " ▁▂▃▄▅▆▇█"
    span = max(t_max - t_min, 1.0)
    if len(values) >= width:
        sample = list(values)[-width:]
    else:
        sample = list(values)
    chars = []
    for v in sample:
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
    from waam_twin.platform import init_taichi

    args = build_parser().parse_args(argv)
    init_taichi()

    from waam_twin import kernels
    from .extract import (
        FILTER_ALL,
        FILTER_LIQUID,
        FILTER_SOLID,
        FILTER_SURFACE,
        extract_body_force,
        extract_flow_arrows,
        extract_flow_arrows_near,
        extract_force_arrows,
        extract_force_arrows_near,
        extract_haz,
        extract_melt_pool,
        extract_surface_points,
        extract_torch_marker,
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
    )
    twin = session.twin
    g = twin.grid
    particle_scale = max(0.05, min(1.0, args.particle_scale))
    tracer_scale = particle_scale * 1.35

    count_field = ti.field(dtype=ti.i32, shape=())
    dx_mm = g.dx * 1000.0
    max_cells = g.nx * g.ny * g.nz
    render_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_cells + 1)
    render_col = ti.Vector.field(3, dtype=ti.f32, shape=max_cells + 1)

    max_arrows = 4096
    arrow_vert = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)
    arrow_col = ti.Vector.field(3, dtype=ti.f32, shape=max_arrows * 2)

    tracer_pos = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)
    tracer_col = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)

    sl_vert = ti.Vector.field(3, dtype=ti.f32, shape=8192)
    sl_col = ti.Vector.field(3, dtype=ti.f32, shape=8192)

    window = ti.ui.Window("WAAM Digital Twin", (1280, 720), vsync=True)
    camera = ti.ui.Camera()

    cx = g.nx * dx_mm / 2.0 + session.offset_x_mm()
    cy = g.ny * dx_mm / 2.0
    cz = g.nz * dx_mm / 2.0

    camera.position(cx * 0.5, -35.0, cz * 2.0)
    camera.lookat(cx, cy, cz * 0.3)
    camera.up(0.0, 0.0, 1.0)

    paused = args.paused
    render_mode = MODE_TEMP
    flow_mode = FLOW_OFF
    steps_per_frame = max(1, args.steps_per_frame)
    show_tracers = True
    show_torch = True
    show_surface_mesh = False
    clip_y = False
    clip_z = False
    filter_mode = FILTER_LIQUID if args.liquid_only else FILTER_ALL
    use_phi = 1 if twin.enable_vof else 0
    dump_idx = 0
    streamline_cache: list[np.ndarray] = []
    streamline_frame = -999
    num_flow_arrows = 0
    pick_cell: tuple[int, int, int] | None = None
    pick_label = ""
    probe_t_history: deque[float] = deque(maxlen=PROBE_HISTORY_LEN)
    probe_spark_min = 300.0
    probe_spark_max = 2000.0

    if twin.probe_recorder is None and getattr(twin, "_job_config", None):
        probes_cfg = twin._job_config.get("probes")
        if probes_cfg:
            from waam_twin.export.probes import ProbeRecorder
            twin.probe_recorder = ProbeRecorder.from_job_list(probes_cfg, twin)

    _print_controls()

    print(f"\n  Job     : {session.job_label}")
    print(f"  Grid    : {g.nx}×{g.ny}×{g.nz}  dx={dx_mm:.3f} mm")
    print(f"  VOF     : {twin.enable_vof}  |  Path: {session.uses_path}")
    print(f"  Travel  : {session.torch_spd_m_s * 1000:.2f} mm/s")
    print(f"  Particles: radius={particle_scale:.2f}×dx  (--particle-scale)")
    print(f"  Output  : {out_dir.resolve()}\n")

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
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
                show_surface_mesh = not show_surface_mesh
                print(f"[viewer] Surface view → {'φ shell (T-colored)' if show_surface_mesh else 'particles'}")
            elif e.key in ("c", "C"):
                clip_y = not clip_y
            elif e.key in ("z", "Z"):
                clip_z = not clip_z
            elif e.key in ("t", "T"):
                show_tracers = not show_tracers
            elif e.key in ("o", "O"):
                show_torch = not show_torch
            elif e.key in ("r", "R"):
                session.reset_motion()
                streamline_cache = []
                probe_t_history.clear()
            elif e.key == "=" or e.key == "+":
                steps_per_frame = min(500, steps_per_frame + 5)
            elif e.key == "-" or e.key == "_":
                steps_per_frame = max(1, steps_per_frame - 5)
            elif e.key in ("s", "S"):
                path = out_dir / f"frame_{dump_idx:05d}.png"
                try:
                    window.save_image(str(path))
                    print(f"[viewer] Screenshot → {path}")
                    dump_idx += 1
                except Exception as exc:
                    print(f"[viewer] Screenshot failed: {exc}")
            elif e.key in ("g", "G"):
                tag = f"step_{twin._step_n:06d}"
                bundle_dir = out_dir / f"bundle_{tag}"
                try:
                    twin.export_research_bundle(
                        bundle_dir,
                        tag=tag,
                        tiers=(0, 1, 2, 3),
                        include_surface=True,
                        include_tracers=True,
                    )
                except Exception as exc:
                    print(f"[viewer] Bundle export failed: {exc}")
            elif e.key in ("p", "P"):
                tx, ty, tz = session.torch_mm()
                if twin.probe_recorder is None:
                    from waam_twin.export.probes import ProbeRecorder
                    twin.probe_recorder = ProbeRecorder()
                g2 = twin.grid
                ti_c = int(np.clip((tx - session.offset_x_mm()) / dx_mm, 0, g2.nx - 1))
                tj_c = int(np.clip(ty / dx_mm, 0, g2.ny - 1))
                tk_c = int(np.clip(tz / dx_mm, 0, g2.nz - 1))
                twin.probe_recorder.add_grid(ti_c, tj_c, tk_c, f"torch_{len(twin.probe_recorder.probes)}")
                print(f"[viewer] Probe added at ({ti_c},{tj_c},{tk_c})")
            elif e.key in ("i", "I"):
                la = camera.curr_lookat
                look_mm = (float(la[0]), float(la[1]), float(la[2]))
                pi, pj, pk = lookat_to_grid(look_mm, twin, session.offset_x_mm())
                pick_cell = (pi, pj, pk)
                pick_label = f"lookat_{pi}_{pj}_{pk}"
                if twin.probe_recorder is None:
                    from waam_twin.export.probes import ProbeRecorder
                    twin.probe_recorder = ProbeRecorder()
                twin.probe_recorder.add_grid(pi, pj, pk, pick_label)
                vals = sample_cell(twin, pi, pj, pk)
                print(
                    f"[viewer] Pick ({pi},{pj},{pk})  T={vals['T_C']:.0f}°C  "
                    f"T_max={vals['T_max_K']:.0f}K  f_l={vals['f_l']:.3f}"
                )
            elif e.key == ti.ui.ESCAPE:
                window.running = False

        if not paused:
            session.advance_physics(steps_per_frame, is_welding=True)

        kernels.compute_vorticity_magnitude(
            g.ux, g.uy, g.uz, g.flags, g.vorticity_mag,
            g.dx, g.dt, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )

        canvas = window.get_canvas()
        scene = window.get_scene()
        # Invert yaw only; pitch uses Taichi default (drag up → look up).
        camera.track_user_inputs(
            window,
            movement_speed=2.0,
            hold_key=ti.ui.LMB,
            yaw_speed=-2.0,
            pitch_speed=2.0,
        )
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(cx, cy - 30.0, cz + 40.0), color=(1.0, 0.95, 0.85))

        offset_x_mm = session.offset_x_mm()
        clip_y_i = 1 if clip_y else 0
        clip_z_i = 1 if clip_z else 0

        reset_count(count_field)
        if show_surface_mesh and twin.enable_vof:
            extract_surface_points(
                g.phi, g.T, g.f_l, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm,
                twin.mat.T_solidus, twin.mat.T_liquidus,
                g.FLAG_GAS, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
        elif render_mode == MODE_TEMP:
            extract_melt_pool(
                g.f_l, g.T, g.T_max, g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm,
                twin.mat.T_solidus, twin.mat.T_liquidus,
                twin.nz_solid,
                g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                filter_mode, use_phi, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
        elif render_mode == MODE_HAZ:
            extract_haz(
                g.T_max, g.f_l, g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm,
                twin.mat.T_solidus, g.FLAG_GAS, g.FLAG_SOLID,
                filter_mode, use_phi, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
        elif render_mode == MODE_VEL:
            extract_velocity(
                g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm, g.dx, g.dt, g.u_ref_phys,
                g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                filter_mode, use_phi, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
        elif render_mode == MODE_VORT:
            extract_vorticity(
                g.vorticity_mag, g.f_l, g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm, 5000.0,
                g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                filter_mode, use_phi, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
        else:
            extract_body_force(
                g.Fx_snap, g.Fy_snap, g.Fz_snap,
                g.f_l, g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm, 0.01,
                g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID,
                filter_mode, use_phi, max_cells,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )

        num_cells = count_field[None]

        if show_torch:
            tx, ty, tz = session.torch_mm()
            extract_torch_marker(
                render_pos, render_col, count_field,
                tx, ty, tz, max_cells + 1,
            )
            num_cells = count_field[None]

        if pick_cell is not None:
            pi, pj, pk = pick_cell
            extract_torch_marker(
                render_pos, render_col, count_field,
                pi * dx_mm + offset_x_mm, pj * dx_mm, pk * dx_mm,
                max_cells + 1,
            )
            num_cells = count_field[None]

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
            tx, ty, tz = session.torch_mm()
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
                    0, 0, g.ny, g.nz,
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
            tx, ty, tz = session.torch_mm()
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

        if flow_mode == FLOW_STREAMLINES and twin._step_n - streamline_frame > 4:
            streamline_frame = twin._step_n
            tx, ty, tz = session.torch_mm()
            ti_c = int(np.clip((tx - offset_x_mm) / dx_mm, 0, g.nx - 1))
            tj_c = int(np.clip(ty / dx_mm, 0, g.ny - 1))
            tk_c = int(np.clip(tz / dx_mm, 0, g.nz - 1))
            seeds = seeds_near_torch(g.nx, g.ny, g.nz, ti_c, tj_c, tk_c, 16)
            streamline_cache = trace_streamlines(
                g.ux.to_numpy(), g.uy.to_numpy(), g.uz.to_numpy(),
                seeds, n_steps=45, step_cells=0.7,
            )

        if flow_mode == FLOW_STREAMLINES and streamline_cache:
            vert_list: list[list[float]] = []
            base = 0
            n_seg = 0
            for line in streamline_cache:
                for k in range(len(line) - 1):
                    if base + 1 >= sl_vert.shape[0]:
                        break
                    p0, p1 = line[k], line[k + 1]
                    sl_vert[base] = [
                        float(p0[0]) * dx_mm + offset_x_mm,
                        float(p0[1]) * dx_mm,
                        float(p0[2]) * dx_mm,
                    ]
                    sl_vert[base + 1] = [
                        float(p1[0]) * dx_mm + offset_x_mm,
                        float(p1[1]) * dx_mm,
                        float(p1[2]) * dx_mm,
                    ]
                    sl_col[base] = [0.15, 0.95, 1.0]
                    sl_col[base + 1] = [0.15, 0.95, 1.0]
                    base += 2
                    n_seg += 1
            if n_seg > 0:
                scene.lines(
                    sl_vert,
                    per_vertex_color=sl_col,
                    width=3.0,
                    vertex_count=n_seg * 2,
                )

        reset_count(count_field)
        extract_tracers(
            g.porosity_pos, g.porosity_active,
            tracer_pos, tracer_col, count_field,
            g.max_tracers, g.max_tracers,
            offset_x_mm, clip_y_i, clip_z_i, g.dx, g.ny, g.nz,
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

        telem = twin.get_telemetry()
        filter_labels = {
            FILTER_ALL: "all metal",
            FILTER_LIQUID: "liquid",
            FILTER_SURFACE: "surface",
            FILTER_SOLID: "solid (HAZ)",
        }
        status_str = "PAUSED" if paused else f"LIVE  {steps_per_frame} st/frame"

        # Probe T(t) for sparkline (first probe or pick cell)
        spark_name = ""
        if twin.probe_recorder and twin.probe_recorder.probes:
            p0 = twin.probe_recorder.probes[0]
            vals = sample_cell(twin, p0.i, p0.j, p0.k)
            probe_t_history.append(vals["T_K"])
            spark_name = p0.name
            probe_spark_min = min(probe_spark_min, vals["T_K"] - 50.0)
            probe_spark_max = max(probe_spark_max, vals["T_K"] + 50.0)
        elif pick_cell is not None:
            vals = sample_cell(twin, *pick_cell)
            probe_t_history.append(vals["T_K"])
            spark_name = pick_label or "pick"

        window.GUI.begin("WAAM Twin", 0.01, 0.01, 0.42, 0.46)
        window.GUI.text(f"Job    : {session.job_label}")
        window.GUI.text(f"Status : {status_str}")
        window.GUI.text(f"View   : {MODE_NAMES[render_mode]}")
        flow_hint = FLOW_NAMES[flow_mode]
        if flow_mode != FLOW_OFF:
            flow_hint += f"  ({num_flow_arrows} arrows)" if flow_mode == FLOW_ARROWS else f"  ({len(streamline_cache)} lines)"
        window.GUI.text(f"Flow   : {flow_hint}  (V=cycle)")
        if flow_mode != FLOW_OFF and num_flow_arrows == 0:
            window.GUI.text("  (no flow arrows at torch — try B=liquid, Z=off)")
        window.GUI.text(f"Filter : {filter_labels.get(filter_mode, '?')}  (B/H/F)")
        window.GUI.text(f"Surface: {'φ shell' if show_surface_mesh else 'particles'}  (N)")
        window.GUI.text(f"Clip Y : {'ON' if clip_y else 'OFF'}  Z : {'ON' if clip_z else 'OFF'}")
        if clip_z:
            window.GUI.text("  (Z clip hides crown — press Z)")
        window.GUI.text(f"Sim t  : {telem['sim_time_ms']:.2f} ms  step {telem['step']}")
        window.GUI.text(
            f"Pool   : W {telem['pool_width_mm']:.2f} mm  "
            f"D {telem['pool_depth_mm']:.2f} mm"
        )
        window.GUI.text(
            f"T_peak : {telem['peak_temp_C']:.0f} °C  "
            f"u_max {telem['marangoni_vel_ms']:.3f} m/s"
        )
        window.GUI.text(
            f"Metal  : {num_cells} rendered  |  "
            f"liquid {telem.get('n_liquid_cells', 0)}"
        )
        bh = telem.get("bead_height_mm", 0.0)
        dep = telem.get("deposited_mass_g", 0.0)
        window.GUI.text(
            f"Bead   : h {bh:.2f} mm  deposited {dep:.3f} g  "
            f"tracers {num_tracers}"
        )
        window.GUI.text(f"Mat    : {telem['material_name']} ({telem['material_status']})")
        if pick_cell is not None:
            pv = sample_cell(twin, *pick_cell)
            window.GUI.text(
                f"Pick   : ({pick_cell[0]},{pick_cell[1]},{pick_cell[2]})  "
                f"T={pv['T_C']:.0f}°C  f_l={pv['f_l']:.2f}"
            )
        if twin.probe_recorder and twin.probe_recorder.probes:
            window.GUI.text(f"Probes : {len(twin.probe_recorder.probes)}  (P=torch  I=lookat)")
        window.GUI.end()

        if probe_t_history:
            window.GUI.begin("T(t) probe", 0.01, 0.50, 0.42, 0.22)
            window.GUI.text(f"Probe  : {spark_name}")
            window.GUI.text(f"T now  : {probe_t_history[-1] - 273.15:.0f} °C")
            window.GUI.text(_sparkline(probe_t_history, probe_spark_min, probe_spark_max))
            window.GUI.text(f"range  : {probe_spark_min - 273.15:.0f}–{probe_spark_max - 273.15:.0f} °C")
            window.GUI.end()

        window.show()


def _print_controls() -> None:
    print("\nControls:")
    print("  SPACE     Pause / resume")
    print("  M         Cycle view (T / HAZ / Velocity / Vorticity / Body force)")
    print("  V         Cycle flow overlay (off / arrows / streamlines)")
    print("  B / H / F Filter: all / solid / surface")
    print("  N         Toggle φ surface shell (T-colored)")
    print("  C / Z     Toggle Y / Z cross-section clip")
    print("  T / O     Toggle tracers / torch marker")
    print("  G         Full research VTK bundle → viewer_output/")
    print("  P         Add probe at torch position")
    print("  I         Pick probe at camera lookat (screen center)")
    print("  R         Reset simulation")
    print("  + / -     More / fewer steps per frame")
    print("  S         Screenshot → viewer_output/")
    print("  LMB drag  Orbit camera   ESC  Exit\n")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
