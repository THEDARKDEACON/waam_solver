"""Taichi GGUI main loop for the WAAM melt-pool viewer."""

from __future__ import annotations

import pathlib

import numpy as np
import taichi as ti

from .cli import build_parser
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


def _ensure_output_dir(path: str | None) -> pathlib.Path:
    if path is None:
        from ..paths import PROJECT_ROOT
        out = PROJECT_ROOT / "viewer_output"
    else:
        out = pathlib.Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    arrow_idx = ti.field(dtype=ti.i32, shape=(max_arrows, 2))

    # Pre-fill line indices
    @ti.kernel
    def _init_arrow_idx():
        for i in range(max_arrows):
            arrow_idx[i, 0] = i * 2
            arrow_idx[i, 1] = i * 2 + 1

    _init_arrow_idx()

    tracer_pos = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)
    tracer_col = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)

    sl_vert = ti.Vector.field(3, dtype=ti.f32, shape=8192)
    sl_col = ti.Vector.field(3, dtype=ti.f32, shape=8192)
    sl_idx = ti.field(dtype=ti.i32, shape=(4096, 2))

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
                ti_c = int(np.clip(tx / dx_mm, 0, g2.nx - 1))
                tj_c = int(np.clip(ty / dx_mm, 0, g2.ny - 1))
                tk_c = int(np.clip(tz / dx_mm, 0, g2.nz - 1))
                twin.probe_recorder.add_grid(ti_c, tj_c, tk_c, f"torch_{len(twin.probe_recorder.probes)}")
                print(f"[viewer] Probe added at ({ti_c},{tj_c},{tk_c})")
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
        camera.track_user_inputs(window, movement_speed=5.0, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(cx, cy - 30.0, cz + 40.0), color=(1.0, 0.95, 0.85))

        offset_x_mm = session.offset_x_mm()
        clip_y_i = 1 if clip_y else 0
        clip_z_i = 1 if clip_z else 0

        reset_count(count_field)
        if show_surface_mesh and twin.enable_vof:
            extract_surface_points(
                g.phi, g.flags,
                render_pos, render_col, count_field,
                dx_mm, offset_x_mm,
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

        if num_cells > 0 and not show_surface_mesh:
            scene.particles(
                render_pos,
                radius=dx_mm * particle_scale,
                per_vertex_color=render_col,
                index_count=num_cells,
            )
        elif show_surface_mesh and num_cells > 0:
            scene.particles(
                render_pos,
                radius=dx_mm * particle_scale * 0.55,
                per_vertex_color=render_col,
                index_count=num_cells,
            )

        num_arrows = 0
        if flow_mode == FLOW_ARROWS:
            reset_count(count_field)
            stride = max(2, int(round(4.0 / particle_scale)))
            extract_flow_arrows(
                g.ux, g.uy, g.uz, g.f_l, g.phi, g.flags,
                arrow_vert, arrow_col, count_field,
                dx_mm, offset_x_mm, g.dx, g.dt, dx_mm * 2.5,
                stride,
                g.FLAG_GAS, g.FLAG_SOLID,
                filter_mode, use_phi, max_arrows,
                clip_y_i, clip_z_i, g.ny, g.nz,
            )
            num_arrows = count_field[None]
            if num_arrows > 0:
                scene.lines(
                    arrow_vert,
                    indices=arrow_idx,
                    per_vertex_color=arrow_col,
                    width=2.0,
                    index_count=num_arrows,
                )

        if flow_mode == FLOW_STREAMLINES and twin._step_n - streamline_frame > 5:
            streamline_frame = twin._step_n
            tx, ty, tz = session.torch_mm()
            ti_c = int(np.clip((tx - offset_x_mm) / dx_mm, 0, g.nx - 1))
            tj_c = int(np.clip(ty / dx_mm, 0, g.ny - 1))
            tk_c = int(np.clip(tz / dx_mm, 0, g.nz - 1))
            seeds = seeds_near_torch(g.nx, g.ny, g.nz, ti_c, tj_c, tk_c, 12)
            streamline_cache = trace_streamlines(
                g.ux.to_numpy(), g.uy.to_numpy(), g.uz.to_numpy(),
                seeds, n_steps=25,
            )

        if flow_mode == FLOW_STREAMLINES and streamline_cache:
            vert_list: list[list[float]] = []
            idx_list: list[list[int]] = []
            base = 0
            for line in streamline_cache:
                for p in line:
                    vert_list.append([
                        float(p[0]) * dx_mm + offset_x_mm,
                        float(p[1]) * dx_mm,
                        float(p[2]) * dx_mm,
                    ])
                for k in range(len(line) - 1):
                    idx_list.append([base + k, base + k + 1])
                base += len(line)
            n_v = min(len(vert_list), sl_vert.shape[0])
            n_l = min(len(idx_list), sl_idx.shape[0])
            if n_l > 0 and n_v > 0:
                for i in range(n_v):
                    sl_vert[i] = vert_list[i]
                    sl_col[i] = [0.2, 0.9, 1.0]
                for i in range(n_l):
                    sl_idx[i, 0] = idx_list[i][0]
                    sl_idx[i, 1] = idx_list[i][1]
                scene.lines(
                    sl_vert,
                    indices=sl_idx,
                    per_vertex_color=sl_col,
                    width=1.5,
                    index_count=n_l,
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

        window.GUI.begin("WAAM Twin", 0.01, 0.01, 0.42, 0.48)
        window.GUI.text(f"Job    : {session.job_label}")
        window.GUI.text(f"Status : {status_str}")
        window.GUI.text(f"View   : {MODE_NAMES[render_mode]}")
        window.GUI.text(f"Flow   : {FLOW_NAMES[flow_mode]}  (V=cycle)")
        window.GUI.text(f"Filter : {filter_labels.get(filter_mode, '?')}  (B/H/F)")
        window.GUI.text(f"Surface: {'mesh' if show_surface_mesh else 'particles'}  (N)")
        window.GUI.text(f"Clip Y : {'ON' if clip_y else 'OFF'}  Z : {'ON' if clip_z else 'OFF'}")
        if clip_z:
            window.GUI.text("  (Z clip hides bead crown — press Z to show)")
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
        if twin.probe_recorder and twin.probe_recorder.probes:
            window.GUI.text(f"Probes : {len(twin.probe_recorder.probes)}  (P=add at torch)")
        window.GUI.end()

        window.show()


def _print_controls() -> None:
    print("\nControls:")
    print("  SPACE     Pause / resume")
    print("  M         Cycle view (T / HAZ / Velocity / Vorticity / Body force)")
    print("  V         Cycle flow overlay (off / arrows / streamlines)")
    print("  B / H / F Filter: all / solid / surface")
    print("  N         Toggle surface mesh (φ shell)")
    print("  C / Z     Toggle Y / Z cross-section clip")
    print("  T / O     Toggle tracers / torch marker")
    print("  G         Full research VTK bundle → viewer_output/")
    print("  P         Add probe at torch position")
    print("  R         Reset simulation")
    print("  + / -     More / fewer steps per frame")
    print("  S         Screenshot → viewer_output/")
    print("  LMB drag  Orbit camera   ESC  Exit\n")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
