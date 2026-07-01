"""Taichi kernels: simulation fields → GGUI particle buffers."""

import taichi as ti

# Cell filter for extract kernels
FILTER_ALL = 0
FILTER_LIQUID = 1
FILTER_SURFACE = 2
FILTER_SOLID = 3


@ti.kernel
def reset_count(count: ti.template()):
    count[None] = 0


@ti.func
def _cell_passes_filter(
    fl: ti.f32,
    phi: ti.f32,
    flag: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
) -> ti.i32:
    """Taichi-safe: single return, no early return in dynamic branches."""
    result = 1
    if filter_mode == FILTER_LIQUID:
        result = 0
        if fl > 0.05:
            result = 1
    elif filter_mode == FILTER_SURFACE:
        result = 0
        if fl > 0.05:
            result = 1
        if use_phi == 1 and phi > 0.05 and phi < 0.95:
            result = 1
    elif filter_mode == FILTER_SOLID:
        result = 0
        if flag != FLAG_GAS and fl < 0.05:
            result = 1
        if flag == FLAG_SOLID:
            result = 1
    return result


@ti.func
def _temperature_color(T: ti.f32, T_solidus: ti.f32, T_liquidus: ti.f32) -> ti.types.vector(3, ti.f32):
    temp_ratio = ti.max(0.0, ti.min(1.0,
        (T - T_solidus) / (T_liquidus - T_solidus + 500.0)
    ))
    return ti.Vector([1.0, 0.35 + temp_ratio * 0.65, temp_ratio * 0.45])


@ti.kernel
def extract_melt_pool(
    f_l: ti.template(),
    T: ti.template(),
    T_max: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    nz_solid: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Extract metal cells; color by T (liquid) or T/T_max (solid bead & HAZ)."""
    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            Tc = T[i, j, k]
            T_peak = T_max[i, j, k]
            # Cold substrate plate (initial nz_solid layers)
            if k < nz_solid:
                col_arr[idx] = ti.Vector([0.22, 0.24, 0.30])
            elif f_l[i, j, k] > 0.05:
                col_arr[idx] = _temperature_color(Tc, T_solidus, T_liquidus)
            elif flags[i, j, k] == FLAG_SOLID:
                # Frozen bead / HAZ: show peak or current temperature
                T_show = ti.max(Tc, T_peak)
                if T_show > T_solidus + 80.0:
                    col_arr[idx] = _temperature_color(T_show, T_solidus, T_liquidus)
                elif T_show > T_solidus:
                    col_arr[idx] = ti.Vector([0.85, 0.55, 0.25])
                elif k >= nz_solid:
                    # Deposited metal (bead crown), cooled
                    col_arr[idx] = ti.Vector([0.55, 0.48, 0.40])
                else:
                    col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])
            elif flags[i, j, k] == FLAG_FLUID:
                col_arr[idx] = ti.Vector([0.45, 0.45, 0.50])
            else:
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])


@ti.kernel
def extract_haz(
    T_max: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    T_solidus: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Extract metal cells colored by peak temperature (HAZ)."""
    for i, j, k in T_max:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            if f_l[i, j, k] >= 0.01:
                col_arr[idx] = ti.Vector([1.0, 0.5, 0.0])
            elif T_max[i, j, k] > 800.0:
                intensity = ti.max(0.0, ti.min(1.0,
                    (T_max[i, j, k] - 800.0) / (T_solidus - 800.0 + 1e-6)
                ))
                col_arr[idx] = ti.Vector([intensity * 0.8, 0.1, 1.0 - intensity * 0.6])
            else:
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])


@ti.kernel
def extract_velocity(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    dx_m: ti.f32,
    dt_s: ti.f32,
    u_ref_phys: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Color liquid metal by physical speed relative to u_ref."""
    inv_u_ref = 1.0 / (u_ref_phys + 1e-9)

    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            if f_l[i, j, k] > 0.05:
                v_lu = ti.sqrt(ux[i, j, k] ** 2 + uy[i, j, k] ** 2 + uz[i, j, k] ** 2)
                v_phys = v_lu * dx_m / dt_s
                intensity = ti.max(0.0, ti.min(1.0, v_phys * inv_u_ref))
                col_arr[idx] = ti.Vector([intensity, 1.0 - intensity, 1.0])
            elif flags[i, j, k] == FLAG_FLUID:
                col_arr[idx] = ti.Vector([0.45, 0.45, 0.50])
            else:
                col_arr[idx] = ti.Vector([0.35, 0.35, 0.40])


@ti.kernel
def extract_tracers(
    pos_in: ti.template(),
    active_in: ti.template(),
    pos_out: ti.template(),
    col_out: ti.template(),
    count: ti.template(),
    max_tracers: ti.i32,
    max_out: ti.i32,
    offset_x_mm: ti.f32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    dx_m: ti.f32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Extract porosity tracers (positions in metres → mm for display)."""
    for p in range(max_tracers):
        act = active_in[p]
        if act > 0:
            pos_mm = pos_in[p] * 1000.0
            pos_mm.x += offset_x_mm
            if clip_y == 1:
                if pos_mm.y / (dx_m * 1000.0) > ny // 2:
                    continue
            if clip_z == 1:
                if pos_mm.z / (dx_m * 1000.0) > nz // 2:
                    continue

            idx = ti.atomic_add(count[None], 1)
            if idx < max_out:
                pos_out[idx] = pos_mm
                if act == 1:
                    col_out[idx] = ti.Vector([0.2, 1.0, 0.2])
                else:
                    col_out[idx] = ti.Vector([1.0, 0.2, 0.2])


@ti.kernel
def extract_torch_marker(
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    torch_x_mm: ti.f32,
    torch_y_mm: ti.f32,
    torch_z_mm: ti.f32,
    max_out: ti.i32,
):
    """Single bright particle at the arc / torch position."""
    idx = ti.atomic_add(count[None], 1)
    if idx < max_out:
        pos_arr[idx] = ti.Vector([torch_x_mm, torch_y_mm, torch_z_mm])
        col_arr[idx] = ti.Vector([1.0, 0.95, 0.2])


@ti.kernel
def extract_vorticity(
    vort: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    vort_ref: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    inv_ref = 1.0 / (vort_ref + 1e-9)
    for i, j, k in vort:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            intensity = ti.max(0.0, ti.min(1.0, vort[i, j, k] * inv_ref))
            col_arr[idx] = ti.Vector([0.2, intensity, 1.0 - intensity])


@ti.kernel
def extract_body_force(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    f_ref: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    inv_ref = 1.0 / (f_ref + 1e-12)
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            fm = ti.sqrt(Fx[i, j, k] ** 2 + Fy[i, j, k] ** 2 + Fz[i, j, k] ** 2)
            intensity = ti.max(0.0, ti.min(1.0, fm * inv_ref))
            col_arr[idx] = ti.Vector([intensity, 0.15, 1.0 - intensity])


@ti.kernel
def extract_flow_arrows(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    vert_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    dx_m: ti.f32,
    dt_s: ti.f32,
    arrow_len_mm: ti.f32,
    stride: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_arrows: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    for i, j, k in ux:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if i % stride != 0 or j % stride != 0 or k % stride != 0:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if f_l[i, j, k] < 0.2:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        v_lu = ti.Vector([ux[i, j, k], uy[i, j, k], uz[i, j, k]])
        vmag = ti.sqrt(v_lu.dot(v_lu))
        if vmag < 1e-6:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_arrows:
            base = idx * 2
            origin = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            direction = v_lu / vmag * arrow_len_mm
            vert_arr[base] = origin
            vert_arr[base + 1] = origin + direction
            c = ti.Vector([0.2, 0.85, 1.0])
            col_arr[base] = c
            col_arr[base + 1] = c


@ti.kernel
def extract_surface_points(
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    FLAG_GAS: ti.i32,
    max_out: ti.i32,
    clip_y: ti.i32,
    clip_z: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Dense surface shell points (φ interface band) for mesh-like view."""
    for i, j, k in phi:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if clip_y == 1 and j > ny // 2:
            continue
        if clip_z == 1 and k > nz // 2:
            continue
        if phi[i, j, k] < 0.35 or phi[i, j, k] > 0.65:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                ti.f32(i) * dx_mm + offset_x_mm,
                ti.f32(j) * dx_mm,
                ti.f32(k) * dx_mm,
            ])
            t = (phi[i, j, k] - 0.35) / 0.3
            col_arr[idx] = ti.Vector([0.3 + t * 0.5, 0.7, 1.0 - t * 0.3])
