"""Kernels: simulation fields → viewer particle buffers."""

from waam_twin.compiler import ti

# Cell filter for extract kernels
FILTER_ALL = 0
FILTER_LIQUID = 1
FILTER_SURFACE = 2
FILTER_SOLID = 3


@ti.func
def _cut_away(i: ti.i32, j: ti.i32, k: ti.i32, cut_on: ti.i32,
              nx: ti.f32, ny: ti.f32, nz: ti.f32, d: ti.f32) -> ti.i32:
    """1 when the cell centre is on the removed side of the section plane."""
    r = 0
    if cut_on == 1:
        if nx * (ti.f32(i) + 0.5) + ny * (ti.f32(j) + 0.5) + nz * (ti.f32(k) + 0.5) - d < 0.0:
            r = 1
    return r


@ti.kernel
def ray_pick_metal(
    phi: ti.template(),
    flags: ti.template(),
    out: ti.template(),
    ox: ti.f32, oy: ti.f32, oz: ti.f32,
    dx_: ti.f32, dy_: ti.f32, dz_: ti.f32,
    max_len: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
    cut_on: ti.i32, cut_nx: ti.f32, cut_ny: ti.f32, cut_nz: ti.f32, cut_d: ti.f32,
):
    """First metal cell (φ ≥ 0.5, kept side of the cut) along a ray.

    Origin and unit direction in window-local cell units. out = (hit, x, y, z)
    with the hit point in cell units (hit = 0 when the ray finds no metal).
    """
    for _ in range(1):
        out[0] = 0.0
        t = 0.0
        found = 0
        while t < max_len and found == 0:
            x = ox + t * dx_
            y = oy + t * dy_
            z = oz + t * dz_
            i = ti.cast(ti.floor(x), ti.i32)
            j = ti.cast(ti.floor(y), ti.i32)
            k = ti.cast(ti.floor(z), ti.i32)
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                kept = 1
                if cut_on == 1 and cut_nx * x + cut_ny * y + cut_nz * z - cut_d < 0.0:
                    kept = 0
                if kept == 1 and flags[i, j, k] != FLAG_GAS and phi[i, j, k] >= 0.5:
                    found = 1
                    out[0] = 1.0
                    out[1] = x
                    out[2] = y
                    out[3] = z
            t += 0.25


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
    """Legacy melt-pool colormap (solidus → liquidus+)."""
    temp_ratio = ti.max(0.0, ti.min(1.0,
        (T - T_solidus) / (T_liquidus - T_solidus + 500.0)
    ))
    return ti.Vector([1.0, 0.35 + temp_ratio * 0.65, temp_ratio * 0.45])


@ti.func
def _thermal_field_color(T: ti.f32, T_lo: ti.f32, T_hi: ti.f32) -> ti.types.vector(3, ti.f32):
    """Full-field heat-transfer colormap: blue → cyan → yellow → red → white.

    Spans ambient → hot so substrate gradients are visible (not flat gray).
    """
    u = ti.max(0.0, ti.min(1.0, (T - T_lo) / (T_hi - T_lo + 1e-3)))
    c = ti.Vector([0.05, 0.08, 0.35])
    if u < 0.25:
        t = u / 0.25
        c = ti.Vector([0.05, 0.08 + 0.55 * t, 0.35 + 0.55 * t])  # blue → cyan
    elif u < 0.5:
        t = (u - 0.25) / 0.25
        c = ti.Vector([0.1 + 0.85 * t, 0.63 + 0.27 * t, 0.9 - 0.7 * t])  # cyan → yellow
    elif u < 0.75:
        t = (u - 0.5) / 0.25
        c = ti.Vector([0.95, 0.9 - 0.55 * t, 0.2 * (1.0 - t)])  # yellow → red
    else:
        t = (u - 0.75) / 0.25
        c = ti.Vector([0.95 + 0.05 * t, 0.35 + 0.55 * t, 0.15 + 0.7 * t])  # red → white-hot
    return c


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
    T_amb: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    nz_solid: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Extract metal cells colored by continuous temperature (heat-transfer field).

    With a section plane (cut_on), only cells on its kept side are drawn
    (n·centre − d ≥ 0, cell units), so the cut face shows the field inside.
    """
    T_hi = T_liquidus + 800.0
    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            # Live T everywhere — shows conduction into the cold plate.
            col_arr[idx] = _thermal_field_color(T[i, j, k], T_amb, T_hi)


@ti.func
def _deposit_color(frac: ti.f32, fl: ti.f32, T: ti.f32, T_solidus: ti.f32, T_hi: ti.f32) -> ti.types.vector(3, ti.f32):
    """Wire metal copper, plate steel-grey, fusion-zone mix in between;
    liquid glows from orange to white with temperature."""
    plate = ti.Vector([0.50, 0.53, 0.58])
    wire = ti.Vector([0.80, 0.50, 0.22])
    w = ti.max(0.0, ti.min(1.0, frac))
    c = wire * (1.0 - w) + plate * w
    if fl > 0.05:
        u = ti.max(0.0, ti.min(1.0, (T - T_solidus) / (T_hi - T_solidus + 1e-3)))
        hot = ti.Vector([1.0, 0.45 + 0.5 * u, 0.1 + 0.8 * u])
        a = ti.min(1.0, 0.4 + fl)
        c = c * (1.0 - a) + hot * a
    return c


@ti.kernel
def extract_deposit(
    alloy_frac: ti.template(),
    f_l: ti.template(),
    T: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Metal cells coloured by birth alloy: the deposit stands out from the
    plate at any temperature (alloy_frac 0 = wire, 1 = plate)."""
    T_hi = T_liquidus + 800.0
    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            col_arr[idx] = _deposit_color(
                alloy_frac[i, j, k], f_l[i, j, k], T[i, j, k], T_solidus, T_hi,
            )


@ti.kernel
def extract_section(
    phi: ti.template(),
    flags: ti.template(),
    out: ti.template(),
    axis: ti.i32,
    idx: ti.i32,
    n_t: ti.i32,
    nz: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Metal fraction on one transverse plane: out[t, k] = phi of the cell
    (0 for gas). axis 0: plane i = idx, t = j; axis 1: plane j = idx, t = i."""
    for t, k in ti.ndrange(n_t, nz):
        i = idx
        j = t
        if axis == 1:
            i = t
            j = idx
        v = 0.0
        if flags[i, j, k] != FLAG_GAS:
            v = ti.max(0.0, ti.min(1.0, phi[i, j, k]))
        out[t, k] = v


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
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Extract metal cells colored by peak temperature (HAZ)."""
    for i, j, k in T_max:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            if f_l[i, j, k] >= 0.01:
                col_arr[idx] = ti.Vector([1.0, 0.5, 0.0])
            # HAZ band starts at 800 °C = 1073.15 K (T_max is in Kelvin —
            # comparing against 800 K = 527 °C overextended the shown HAZ).
            elif T_max[i, j, k] > 1073.15:
                intensity = ti.max(0.0, ti.min(1.0,
                    (T_max[i, j, k] - 1073.15) / (T_solidus - 1073.15 + 1e-6)
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
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Color liquid metal by physical speed relative to u_ref."""
    inv_u_ref = 1.0 / (u_ref_phys + 1e-9)

    for i, j, k in f_l:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue

        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
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
    dx_m: ti.f32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Extract porosity tracers (positions in metres → mm for display)."""
    for p in range(max_tracers):
        act = active_in[p]
        if act > 0:
            pos_mm = pos_in[p] * 1000.0
            pos_mm.x += offset_x_mm
            c = pos_in[p] / dx_m  # window-local cell units
            if cut_on == 1 and cut_nx * c.x + cut_ny * c.y + cut_nz * c.z - cut_d < 0.0:
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
    """Yellow contact-tip marker (surface + CTWD)."""
    idx = ti.atomic_add(count[None], 1)
    if idx < max_out:
        pos_arr[idx] = ti.Vector([torch_x_mm, torch_y_mm, torch_z_mm])
        col_arr[idx] = ti.Vector([1.0, 0.95, 0.2])


@ti.kernel
def extract_arc_attach_marker(
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    x_mm: ti.f32,
    y_mm: ti.f32,
    z_mm: ti.f32,
    max_out: ti.i32,
):
    """Cyan marker at the free-surface arc attachment (where heat is injected)."""
    idx = ti.atomic_add(count[None], 1)
    if idx < max_out:
        pos_arr[idx] = ti.Vector([x_mm, y_mm, z_mm])
        col_arr[idx] = ti.Vector([0.15, 0.95, 1.0])


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
    vort_floor: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_out: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Color liquid/mush by |ω|; hide near-zero solid so the pool stands out."""
    inv_ref = 1.0 / (vort_ref + 1e-9)
    for i, j, k in vort:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        v = vort[i, j, k]
        # Skip stagnant cells — otherwise the whole plate is flat dark blue.
        if v < vort_floor:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            # Log-ish: map [floor, ref] → [0,1] then purple→yellow
            intensity = ti.max(0.0, ti.min(1.0, ti.log(1.0 + v * inv_ref * 9.0) / ti.log(10.0)))
            col_arr[idx] = ti.Vector([
                0.35 + 0.65 * intensity,
                0.15 + 0.75 * intensity,
                0.95 - 0.75 * intensity,
            ])


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
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    inv_ref = 1.0 / (f_ref + 1e-12)
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
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
    min_len_mm: ti.f32,
    stride: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_arrows: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Velocity arrow segments (origin → tip) in mm; length scales with |u|."""
    u_ref = dx_m / dt_s + 1e-9
    for i, j, k in ux:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if i % stride != 0 or j % stride != 0 or k % stride != 0:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        fl = f_l[i, j, k]
        if fl < 0.08 and not (use_phi == 1 and phi[i, j, k] > 0.15 and phi[i, j, k] < 0.95):
            continue
        if _cell_passes_filter(
            fl, phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        v_lu = ti.Vector([ux[i, j, k], uy[i, j, k], uz[i, j, k]])
        vmag = ti.sqrt(v_lu.dot(v_lu))
        if vmag < 1e-12:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_arrows:
            base = idx * 2
            origin = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            v_phys = vmag * u_ref
            seg_len = ti.max(min_len_mm, ti.min(arrow_len_mm, v_phys * arrow_len_mm * 4.0))
            direction = (v_lu / vmag) * seg_len
            vert_arr[base] = origin
            vert_arr[base + 1] = origin + direction
            intensity = ti.max(0.25, ti.min(1.0, v_phys / (u_ref * 0.15 + 1e-9)))
            c = ti.Vector([0.15, 0.55 + intensity * 0.45, 1.0])
            col_arr[base] = c
            col_arr[base + 1] = c


@ti.kernel
def extract_flow_arrows_near(
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
    min_len_mm: ti.f32,
    ci: ti.i32,
    cj: ti.i32,
    ck: ti.i32,
    radius: ti.i32,
    sub_stride: ti.i32,
    FLAG_GAS: ti.i32,
    max_arrows: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Velocity arrows in a dense box around the torch (ignores global stride/clip)."""
    u_ref = dx_m / dt_s + 1e-9
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for dk in range(-radius, radius + 1):
                if di % sub_stride != 0 or dj % sub_stride != 0 or dk % sub_stride != 0:
                    continue
                i = ci + di
                j = cj + dj
                k = ck + dk
                if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                    continue
                if flags[i, j, k] == FLAG_GAS:
                    continue
                fl = f_l[i, j, k]
                if fl < 0.15 and phi[i, j, k] < 0.2:
                    continue
                v_lu = ti.Vector([ux[i, j, k], uy[i, j, k], uz[i, j, k]])
                vmag = ti.sqrt(v_lu.dot(v_lu))
                if vmag < 1e-12:
                    continue
                idx = ti.atomic_add(count[None], 1)
                if idx < max_arrows:
                    base = idx * 2
                    origin = ti.Vector([
                        (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                        (ti.f32(j) + 0.5) * dx_mm,
                        (ti.f32(k) + 0.5) * dx_mm,
                    ])
                    v_phys = vmag * u_ref
                    seg_len = ti.max(min_len_mm, ti.min(arrow_len_mm, v_phys * arrow_len_mm * 4.0))
                    direction = (v_lu / vmag) * seg_len
                    vert_arr[base] = origin
                    vert_arr[base + 1] = origin + direction
                    intensity = ti.max(0.25, ti.min(1.0, v_phys / (u_ref * 0.15 + 1e-9)))
                    c = ti.Vector([0.15, 0.55 + intensity * 0.45, 1.0])
                    col_arr[base] = c
                    col_arr[base + 1] = c


@ti.kernel
def extract_force_arrows_near(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    vert_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    arrow_len_mm: ti.f32,
    f_ref_lu: ti.f32,
    ci: ti.i32,
    cj: ti.i32,
    ck: ti.i32,
    radius: ti.i32,
    sub_stride: ti.i32,
    FLAG_GAS: ti.i32,
    max_arrows: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for dk in range(-radius, radius + 1):
                if di % sub_stride != 0 or dj % sub_stride != 0 or dk % sub_stride != 0:
                    continue
                i = ci + di
                j = cj + dj
                k = ck + dk
                if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                    continue
                if flags[i, j, k] == FLAG_GAS:
                    continue
                if f_l[i, j, k] < 0.1:
                    continue
                f_lu = ti.Vector([Fx[i, j, k], Fy[i, j, k], Fz[i, j, k]])
                fmag = ti.sqrt(f_lu.dot(f_lu))
                if fmag < 1e-12:
                    continue
                idx = ti.atomic_add(count[None], 1)
                if idx < max_arrows:
                    base = idx * 2
                    origin = ti.Vector([
                        (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                        (ti.f32(j) + 0.5) * dx_mm,
                        (ti.f32(k) + 0.5) * dx_mm,
                    ])
                    seg_len = ti.max(dx_mm * 0.8, ti.min(arrow_len_mm, fmag / f_ref_lu * arrow_len_mm))
                    direction = (f_lu / fmag) * seg_len
                    vert_arr[base] = origin
                    vert_arr[base + 1] = origin + direction
                    intensity = ti.max(0.2, ti.min(1.0, fmag / (f_ref_lu + 1e-12)))
                    c = ti.Vector([intensity, 0.2, 1.0 - intensity * 0.7])
                    col_arr[base] = c
                    col_arr[base + 1] = c


@ti.kernel
def extract_force_arrows(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    vert_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    arrow_len_mm: ti.f32,
    f_ref_lu: ti.f32,
    stride: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    filter_mode: ti.i32,
    use_phi: ti.i32,
    max_arrows: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Body-force arrow segments (same layout as velocity arrows)."""
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if i % stride != 0 or j % stride != 0 or k % stride != 0:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if f_l[i, j, k] < 0.08:
            continue
        if _cell_passes_filter(
            f_l[i, j, k], phi[i, j, k], flags[i, j, k],
            filter_mode, use_phi, FLAG_SOLID, FLAG_GAS,
        ) == 0:
            continue
        f_lu = ti.Vector([Fx[i, j, k], Fy[i, j, k], Fz[i, j, k]])
        fmag = ti.sqrt(f_lu.dot(f_lu))
        if fmag < 1e-12:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_arrows:
            base = idx * 2
            origin = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            seg_len = ti.max(dx_mm * 0.8, ti.min(arrow_len_mm, fmag / f_ref_lu * arrow_len_mm))
            direction = (f_lu / fmag) * seg_len
            vert_arr[base] = origin
            vert_arr[base + 1] = origin + direction
            intensity = ti.max(0.2, ti.min(1.0, fmag / (f_ref_lu + 1e-12)))
            c = ti.Vector([intensity, 0.2, 1.0 - intensity * 0.7])
            col_arr[base] = c
            col_arr[base + 1] = c


@ti.kernel
def extract_surface_points(
    phi: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    pos_arr: ti.template(),
    col_arr: ti.template(),
    count: ti.template(),
    dx_mm: ti.f32,
    offset_x_mm: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    FLAG_GAS: ti.i32,
    max_out: ti.i32,
    cut_on: ti.i32,
    cut_nx: ti.f32,
    cut_ny: ti.f32,
    cut_nz: ti.f32,
    cut_d: ti.f32,
):
    """Dense φ interface shell; color by local temperature."""
    for i, j, k in phi:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if _cut_away(i, j, k, cut_on, cut_nx, cut_ny, cut_nz, cut_d) == 1:
            continue
        if phi[i, j, k] < 0.35 or phi[i, j, k] > 0.65:
            continue
        idx = ti.atomic_add(count[None], 1)
        if idx < max_out:
            pos_arr[idx] = ti.Vector([
                (ti.f32(i) + 0.5) * dx_mm + offset_x_mm,
                (ti.f32(j) + 0.5) * dx_mm,
                (ti.f32(k) + 0.5) * dx_mm,
            ])
            if f_l[i, j, k] > 0.1:
                col_arr[idx] = _temperature_color(T[i, j, k], T_solidus, T_liquidus)
            else:
                t = (phi[i, j, k] - 0.35) / 0.3
                col_arr[idx] = ti.Vector([0.3 + t * 0.5, 0.7, 1.0 - t * 0.3])
