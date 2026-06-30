"""
viewer.py — Real-time GPU LBM Visualization
=============================================
Interactive Taichi GGUI visualizer for the WAAM Digital Twin.
Renders the melt pool topography, porosity tracers, and HAZ in real-time.

Fixes applied (2026-06-28):
  - ti.init() moved inside main() to prevent double-init on import
  - Broken local atomic counter replaced with ti.field(shape=()) shared counters
  - Out-of-bounds guard added to all extract kernels
  - get_canvas / get_scene fetched each frame (GGUI best practice)
  - numpy import removed (unused)
  - CPU fallback added for non-CUDA machines
  - arc_pressure_pa bumped to 100 kPa for visible surface deformation

Run:
    python -m waam_twin.viewer
"""

import taichi as ti

# ──────────────────────────────────────────────────────────────────────────────
#  Module-level shared GPU counters for extract kernels.
#  These are DECLARED here (no GPU memory yet) and ALLOCATED inside main()
#  after ti.init(). They are module-level so the @ti.kernel functions below
#  can reference them directly without closures.
# ──────────────────────────────────────────────────────────────────────────────
_count_field = None   # ti.field(dtype=ti.i32, shape=()) — set in main()


# ──────────────────────────────────────────────────────────────────────────────
#  Rendering Kernels
#  NOTE: Kernel definitions at module scope are fine — they are JIT-compiled
#  and do NOT allocate GPU memory. Only ti.field() calls allocate memory,
#  and those happen inside main() after ti.init().
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def _reset_count(count: ti.template()):
    count[None] = 0


@ti.kernel
def extract_melt_pool(
    f_l:   ti.template(),
    T:     ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count:   ti.template(),   # ti.field(dtype=ti.i32, shape=()) — SHARED counter
    dx_mm: ti.f32,
    T_solidus:  ti.f32,
    T_liquidus: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    ny: ti.i32,
):
    """
    Extract cells into flat arrays for GGUI particle rendering.
    Renders ALL metal (solid + liquid) so the user can see the substrate.
    """
    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
            
        # Optimization: only render surface cells or liquid cells to save fill-rate,
        # but for a 128x64x32 grid, rendering the whole block (~200k particles max) is trivial.
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:   # Bounds guard — prevents OOB write
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            if f_l[i, j, k] > 0.05:
                temp_ratio = ti.max(0.0, ti.min(1.0,
                    (T[i, j, k] - T_solidus) / (T_liquidus - T_solidus + 500.0)
                ))
                # Red-to-yellow gradient: cold liquid = orange, peak = white-yellow
                col_arr[idx] = ti.Vector([1.0, 0.4 + temp_ratio * 0.6, temp_ratio * 0.4])
            elif flags[i, j, k] == FLAG_FLUID:
                col_arr[idx] = ti.Vector([0.45, 0.45, 0.50])  # Slightly lighter/bluish grey for bead
            else:
                # Cold solid metal (original substrate)
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])  # Slate gray


@ti.kernel
def extract_haz(
    T_max: ti.template(),
    f_l:   ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count:   ti.template(),   # ti.field(dtype=ti.i32, shape=())
    dx_mm: ti.f32,
    T_solidus: ti.f32,
    FLAG_GAS: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    ny: ti.i32,
):
    """Extract all metal cells, color-coding the HAZ based on T_max."""
    for i, j, k in T_max:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
            
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            
            if f_l[i, j, k] >= 0.01:
                # Currently liquid/mushy
                col_arr[idx] = ti.Vector([1.0, 0.5, 0.0])
            elif T_max[i, j, k] > 800.0:
                # Solid HAZ
                intensity = ti.max(0.0, ti.min(1.0,
                    (T_max[i, j, k] - 800.0) / (T_solidus - 800.0 + 1e-6)
                ))
                # Blue (cool HAZ) → Purple (near-melt HAZ)
                col_arr[idx] = ti.Vector([intensity * 0.8, 0.1, 1.0 - intensity * 0.6])
            else:
                # Cold solid metal
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])


@ti.kernel
def extract_velocity(
    ux:    ti.template(),
    uy:    ti.template(),
    uz:    ti.template(),
    f_l:   ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count:   ti.template(),
    dx_mm: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    ny: ti.i32,
):
    """Extract metal cells, coloring the liquid pool by its physical velocity magnitude (Marangoni flow)."""
    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
            
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            
            if f_l[i, j, k] > 0.05:
                # Calculate velocity magnitude in lattice units
                v_mag = ti.sqrt(ux[i,j,k]**2 + uy[i,j,k]**2 + uz[i,j,k]**2)
                # Map magnitude to a cyan -> blue -> pink gradient to show flow currents
                intensity = ti.max(0.0, ti.min(1.0, v_mag * 50.0))  # Tuned scaling factor for visibility
                col_arr[idx] = ti.Vector([intensity, 1.0 - intensity, 1.0])
            elif flags[i, j, k] == FLAG_FLUID:
                col_arr[idx] = ti.Vector([0.45, 0.45, 0.50])
            else:
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])


@ti.kernel
def extract_tracers(
    pos_in:   ti.template(),
    active_in: ti.template(),
    pos_out:  ti.template(),
    col_out:  ti.template(),
    count:    ti.template(),   # ti.field(dtype=ti.i32, shape=())
    max_tracers: ti.i32,
    max_out:  ti.i32,
    dx:       ti.f32,
    clip_y:   ti.i32,
    ny:       ti.i32,
):
    """Extract active/trapped tracers for particle rendering."""
    for p in range(max_tracers):
        act = active_in[p]
        if act > 0:
            if clip_y == 1:
                idx_y = pos_in[p].y / dx
                if idx_y > ny // 2:
                    continue
                    
            idx = ti.atomic_add(count[None], 1)
            if idx < max_out:
                # Scale from meters to millimeters
                pos_out[idx] = pos_in[p] * 1000.0
                if act == 1:
                    col_out[idx] = ti.Vector([0.2, 1.0, 0.2])   # Green  — flowing
                else:
                    col_out[idx] = ti.Vector([0.0, 1.0, 1.0])   # Cyan   — trapped


# ──────────────────────────────────────────────────────────────────────────────
#  Main GUI Loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    from waam_twin.platform import init_taichi

    init_taichi()

    from waam_twin import WAAMTwin

    global _count_field
    _count_field = ti.field(dtype=ti.i32, shape=())

    print("=" * 60)
    print("  WAAM Digital Twin v2 — GPU Visualizer")
    print("=" * 60)

    twin = WAAMTwin.from_preset(
        "standard",
        material="materials/placeholders/ER70S-6.yaml",
        arc_power_W=3500.0,
        arc_efficiency=0.8,
        T_ambient=300.0,
        arc_pressure_pa=100_000.0,
        droplet_freq_hz=40.0,
    )
    twin.reset()
    g = twin.grid

    # Use millimeters for rendering to avoid the Camera's near-clipping plane (0.1 units)
    dx_mm = g.dx * 1000.0

    # ── Render buffers ────────────────────────────────────────────────────────
    max_cells = g.nx * g.ny * g.nz
    render_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_cells)
    render_col = ti.Vector.field(3, dtype=ti.f32, shape=max_cells)

    tracer_pos = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)
    tracer_col = ti.Vector.field(3, dtype=ti.f32, shape=g.max_tracers)

    # ── GGUI Window ───────────────────────────────────────────────────────────
    window = ti.ui.Window("WAAM Digital Twin", (1280, 720), vsync=True)
    camera = ti.ui.Camera()

    # Center camera on domain middle, slightly behind and above (coordinates in mm)
    cx = g.nx * dx_mm / 2.0
    cy = g.ny * dx_mm / 2.0
    cz = g.nz * dx_mm / 2.0

    camera.position(cx * 0.5, -35.0, cz * 2.0)
    camera.lookat(cx, cy, cz * 0.3)
    camera.up(0.0, 0.0, 1.0)

    # ── UI state ──────────────────────────────────────────────────────────────
    paused          = False
    render_mode     = 0        # 0 = Melt Pool+Tracers, 1 = HAZ, 2 = Velocity
    steps_per_frame = 20
    show_tracers    = True
    clip_y          = False

    # Torch traversal state (physics is still in meters)
    torch_x   = 0.005
    torch_y   = g.ny * g.dx / 2.0
    torch_spd = 0.004  # 4 mm/s forward travel speed

    print("\nControls:")
    print("  SPACE : Pause / Resume")
    print("  M     : Toggle Mode (Melt Pool / HAZ / Velocity Flow)")
    print("  C     : Toggle Cross-Section Clipping Plane")
    print("  T     : Toggle tracer particle overlay")
    print("  LMB + Drag / W/S/A/D : Move Camera")
    print("  ESC   : Exit\n")

    # ── Main render loop ──────────────────────────────────────────────────────
    while window.running:

        # Input
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key in ('m', 'M'):
                render_mode = (render_mode + 1) % 3
            elif e.key in ('c', 'C'):
                clip_y = not clip_y
            elif e.key in ('t', 'T'):
                show_tracers = not show_tracers
            elif e.key == ti.ui.ESCAPE:
                window.running = False

        # Physics advance
        if not paused:
            for _ in range(steps_per_frame):
                torch_x += torch_spd * g.dt
                if torch_x > (g.nx * g.dx - 0.005) or torch_x < 0.005:
                    torch_spd *= -1.0
                twin.step(torch_x_m=torch_x, torch_y_m=torch_y, is_welding=True)

        # ── GGUI objects fetched each frame (GGUI best practice) ─────────────
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera.track_user_inputs(window, movement_speed=5.0, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(cx, cy - 30.0, cz + 40.0), color=(1.0, 0.95, 0.85))

        # ── Extract fields → render buffers (shared counter, proper atomic) ──
        _reset_count(_count_field)

        clip_val = 1 if clip_y else 0
        if render_mode == 0:
            extract_melt_pool(
                g.f_l, g.T, g.flags,
                render_pos, render_col, _count_field,
                dx_mm,
                twin.mat.T_solidus, twin.mat.T_liquidus,
                g.FLAG_GAS, g.FLAG_FLUID, max_cells, clip_val, g.ny
            )
        elif render_mode == 1:
            extract_haz(
                g.T_max, g.f_l, g.flags,
                render_pos, render_col, _count_field,
                dx_mm, twin.mat.T_solidus, g.FLAG_GAS, max_cells, clip_val, g.ny
            )
        else:
            extract_velocity(
                g.ux, g.uy, g.uz, g.f_l, g.flags,
                render_pos, render_col, _count_field,
                dx_mm, g.FLAG_GAS, g.FLAG_FLUID, max_cells, clip_val, g.ny
            )

        num_cells = _count_field[None]

        # Extract tracers (own counter pass)
        _reset_count(_count_field)
        extract_tracers(
            g.porosity_pos, g.porosity_active,
            tracer_pos, tracer_col, _count_field,
            g.max_tracers, g.max_tracers, g.dx, clip_val, g.ny
        )
        num_tracers = _count_field[None]

        # ── Draw ─────────────────────────────────────────────────────────────
        if num_cells > 0:
            scene.particles(
                render_pos, radius=dx_mm * 0.5,
                per_vertex_color=render_col,
                index_count=num_cells,
            )

        if show_tracers and num_tracers > 0:
            scene.particles(
                tracer_pos, radius=dx_mm * 0.7,
                per_vertex_color=tracer_col,
                index_count=num_tracers,
            )

        canvas.scene(scene)

        # ── 2D HUD overlay ────────────────────────────────────────────────────
        if render_mode == 0:
            mode_str = "Melt Pool + Tracers"
        elif render_mode == 1:
            mode_str = "HAZ Map"
        else:
            mode_str = "Velocity Flow"
            
        clip_str = "ON" if clip_y else "OFF"
        status_str = "PAUSED" if paused else f"LIVE  {steps_per_frame} steps/frame"
        sim_ms     = twin._step_n * g.dt * 1000.0

        window.GUI.begin("WAAM Twin", 0.01, 0.01, 0.35, 0.20)
        window.GUI.text(f"Status : {status_str}")
        window.GUI.text(f"View   : {mode_str}")
        window.GUI.text(f"Clip Y : {clip_str}  (C to toggle)")
        window.GUI.text(f"Sim t  : {sim_ms:.2f} ms")
        window.GUI.text(f"Cells  : {num_cells}")
        window.GUI.text(f"Tracers: {num_tracers}  (T to toggle)")
        window.GUI.end()

        window.show()


if __name__ == "__main__":
    main()
