"""Taichi kernels — grid init, feed, window, tracers."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    ALLOY_PLATE,
    ALLOY_WIRE,
)

# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 1 — Grid Initialisation
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def init_grid(
    f:     ti.template(),   # Distribution field to initialise
    rho:   ti.template(),
    ux:    ti.template(),
    uy:    ti.template(),
    uz:    ti.template(),
    T:     ti.template(),
    H:     ti.template(),
    f_l:   ti.template(),
    phi:   ti.template(),
    flags: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
    T_ambient: ti.f32,
    rho0:      ti.f32,
    cp_rho:    ti.f32,      # wire ρ·cp  [J/(m³·K)]
    cp_rho_plate: ti.f32,   # plate ρ·cp (same as wire unless dual-alloy)
    nz_solid:  ti.i32,      # Number of z-layers that are solid substrate
    plate_i0:  ti.i32,      # Plate footprint [i0,i1) × [j0,j1)
    plate_i1:  ti.i32,
    plate_j0:  ti.i32,
    plate_j1:  ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Quiescent ambient initial state.

    Substrate occupies k < nz_solid only inside [plate_i0, plate_i1) ×
    [plate_j0, plate_j1). Outside that XY footprint those layers are gas
    so coupon side faces can convect/radiate. nz_solid < 0 → all-fluid test.
    """
    for i, j, k in rho:
        for q in ti.static(range(19)):
            f[q, i, j, k] = W[q] * rho0   # Equilibrium at rest → f_eq = w * rho

    for i, j, k in rho:
        rho[i, j, k] = rho0
        ux[i, j, k]  = 0.0
        uy[i, j, k]  = 0.0
        uz[i, j, k]  = 0.0
        T[i, j, k]   = T_ambient
        # f_l represents liquid fraction. Initially everything is solid (0.0) or gas.
        f_l[i, j, k] = 0.0
        alloy_id[i, j, k] = ALLOY_WIRE
        alloy_frac[i, j, k] = 0.0
        if nz_solid < 0:
            flags[i, j, k] = FLAG_FLUID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 1.0
            H[i, j, k] = cp_rho * T_ambient
        else:
            in_plate = (
                i >= plate_i0 and i < plate_i1
                and j >= plate_j0 and j < plate_j1
            )
            if in_plate and k < nz_solid:
                phi[i, j, k] = 1.0
                flags[i, j, k] = FLAG_SOLID
                alloy_id[i, j, k] = ALLOY_PLATE
                alloy_frac[i, j, k] = 1.0
                H[i, j, k] = cp_rho_plate * T_ambient
            else:
                phi[i, j, k] = 0.0
                flags[i, j, k] = FLAG_GAS
                H[i, j, k] = cp_rho * T_ambient


@ti.kernel
def init_aux_fields(
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    Fx_snap: ti.template(),
    Fy_snap: ti.template(),
    Fz_snap: ti.template(),
    Fx:    ti.template(),
    Fy:    ti.template(),
    Fz:    ti.template(),
    T_ambient: ti.f32,
    porosity_active: ti.template(),
    tracer_head:     ti.template(),
    max_tracers:     ti.i32,
):
    """Reset HAZ, body forces, thermal aux fields, and tracer pool."""
    tracer_head[None] = 0
    for p in range(max_tracers):
        porosity_active[p] = 0
    for i, j, k in T_max:
        T_max[i, j, k] = T_ambient
        T_prev[i, j, k] = T_ambient
        dT_dt[i, j, k] = 0.0
        time_above_800[i, j, k] = 0.0
        time_above_1100[i, j, k] = 0.0
        time_above_solidus[i, j, k] = 0.0
        Fx[i, j, k] = 0.0
        Fy[i, j, k] = 0.0
        Fz[i, j, k] = 0.0
        Fx_snap[i, j, k] = 0.0
        Fy_snap[i, j, k] = 0.0
        Fz_snap[i, j, k] = 0.0


@ti.kernel
def clear_forces(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
):
    for i, j, k in Fx:
        Fx[i, j, k] = 0.0
        Fy[i, j, k] = 0.0
        Fz[i, j, k] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 1.5 — Wire Mass Addition (Droplet Feed)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def feed_wire(
    f_src: ti.template(),
    flags: ti.template(),
    f_l:   ti.template(),
    phi:   ti.template(),
    H:     ti.template(),
    T:     ti.template(),
    rho:   ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    droplet_radius: ti.f32,
    target_vol: ti.f32,
    T_drop: ti.f32,
    cp_rho: ti.f32,
    L_rho:  ti.f32,
    rho0:   ti.f32,
    vol_acc: ti.template(),
    cell_vol: ti.f32,
    FLAG_GAS:   ti.i32,
    FLAG_FLUID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Deposit one wire droplet: convert gas cells to liquid until ``target_vol``
    is reached (ṁ / f_drop / ρ). Search uses the nominal sphere plus a taller
    column above the pool when the pool intersects the drop volume.

    Volume budget uses reserve-then-commit atomics: a thread first reserves
    cell_vol on the accumulator and only converts the cell if the reservation
    fit inside target_vol, so concurrent threads cannot overshoot the budget.
    """
    search_r = ti.max(droplet_radius * 2.5, droplet_radius + 2.0)
    i_min = ti.max(0, ti.cast(arc_i - search_r, ti.i32))
    i_max = ti.min(nx, ti.cast(arc_i + search_r, ti.i32) + 1)
    j_min = ti.max(0, ti.cast(arc_j - search_r, ti.i32))
    j_max = ti.min(ny, ti.cast(arc_j + search_r, ti.i32) + 1)
    k_min = ti.max(0, ti.cast(arc_k + 1.0, ti.i32))
    k_max = ti.min(nz, ti.cast(arc_k + search_r * 3.0, ti.i32) + 1)
    drop_cz = arc_k + droplet_radius + 1.0

    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
        if vol_acc[None] >= target_vol:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - drop_cz
        r2 = di * di + dj * dj + dk * dk
        r_ij2 = di * di + dj * dj
        in_sphere = r2 <= droplet_radius * droplet_radius
        in_column = r_ij2 <= search_r * search_r and ti.f32(k) >= arc_k + 1.0
        if (in_sphere or in_column) and flags[i, j, k] == FLAG_GAS:
            reserved = ti.atomic_add(vol_acc[None], cell_vol)
            if reserved >= target_vol:
                ti.atomic_sub(vol_acc[None], cell_vol)
            else:
                flags[i, j, k] = FLAG_FLUID
                f_l[i, j, k] = 1.0
                phi[i, j, k] = 1.0
                T[i, j, k] = T_drop
                H[i, j, k] = cp_rho * T_drop + L_rho
                rho[i, j, k] = rho0
                for q in ti.static(range(19)):
                    f_src[q, i, j, k] = W[q] * rho0


@ti.func
def _has_metal_neighbor(
    phi: ti.template(),
    flags: ti.template(),
    f_l: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
) -> ti.i32:
    """True if a 6-connected neighbour carries metal (fluid, solid, or φ > 0.35)."""
    found = 0
    for di, dj, dk in ti.static([
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]):
        ii = i + di
        jj = j + dj
        kk = k + dk
        in_bounds = ii >= 0 and jj >= 0 and kk >= 0 and ii < nx and jj < ny and kk < nz
        if in_bounds:
            if flags[ii, jj, kk] == FLAG_FLUID or flags[ii, jj, kk] == FLAG_SOLID:
                found = 1
            elif phi[ii, jj, kk] > 0.35 or f_l[ii, jj, kk] > 0.2:
                found = 1
    return found


@ti.kernel
def feed_wire_surface(
    f_src: ti.template(),
    flags: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    H: ti.template(),
    T: ti.template(),
    rho: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    footprint_r: ti.f32,
    droplet_radius: ti.f32,
    target_vol: ti.f32,
    T_drop: ti.f32,
    cp_rho: ti.f32,
    L_rho: ti.f32,
    rho0: ti.f32,
    vol_acc: ti.template(),
    real_acc: ti.template(),
    cell_vol: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Deposit wire droplet on the pool footprint only (no vertical gas column).

    Gas cells must be 6-connected to existing metal and lie within a horizontal
    footprint around the arc; vertical extent is limited to ~footprint above arc_k.

    ``vol_acc`` is the shared reserve-then-commit budget (bounds total work);
    ``real_acc`` counts only genuinely converted gas→fluid volume, so the pass-3
    thermal top-up on already-liquid cells is NOT booked as deposited mass.
    """
    search_r = ti.max(footprint_r, droplet_radius)
    i_min = ti.max(0, ti.cast(arc_i - search_r, ti.i32))
    i_max = ti.min(nx, ti.cast(arc_i + search_r, ti.i32) + 1)
    j_min = ti.max(0, ti.cast(arc_j - search_r, ti.i32))
    j_max = ti.min(ny, ti.cast(arc_j + search_r, ti.i32) + 1)
    k_lo = ti.max(0, ti.cast(arc_k - 1.0, ti.i32))
    k_span = ti.max(droplet_radius + 3.0, 4.0)
    k_hi = ti.min(nz, ti.cast(arc_k + k_span, ti.i32) + 1)
    drop_cz = arc_k + droplet_radius + 0.5
    r_drop2 = droplet_radius * droplet_radius
    foot2 = search_r * search_r

    # Pass 1: gas cells adjacent to pool, near drop centre (preferred surface entry)
    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_lo, k_hi)):
        if vol_acc[None] >= target_vol:
            continue
        if flags[i, j, k] != FLAG_GAS:
            continue
        if _has_metal_neighbor(phi, flags, f_l, i, j, k, FLAG_FLUID, FLAG_SOLID, nx, ny, nz) == 0:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - drop_cz
        r2 = di * di + dj * dj + dk * dk
        r_ij2 = di * di + dj * dj
        if r_ij2 > foot2 and r2 > r_drop2:
            continue
        reserved = ti.atomic_add(vol_acc[None], cell_vol)
        if reserved >= target_vol:
            ti.atomic_sub(vol_acc[None], cell_vol)
        else:
            flags[i, j, k] = FLAG_FLUID
            f_l[i, j, k] = 1.0
            phi[i, j, k] = 1.0
            T[i, j, k] = T_drop
            H[i, j, k] = cp_rho * T_drop + L_rho
            rho[i, j, k] = rho0
            ti.atomic_add(real_acc[None], cell_vol)
            for q in ti.static(range(19)):
                f_src[q, i, j, k] = W[q] * rho0

    # Pass 2: widen footprint if volume short (still no vertical column)
    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_lo, k_hi)):
        if vol_acc[None] >= target_vol:
            continue
        if flags[i, j, k] != FLAG_GAS:
            continue
        if _has_metal_neighbor(phi, flags, f_l, i, j, k, FLAG_FLUID, FLAG_SOLID, nx, ny, nz) == 0:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r_ij2 = di * di + dj * dj
        if r_ij2 > foot2 * 1.44:
            continue
        reserved = ti.atomic_add(vol_acc[None], cell_vol)
        if reserved >= target_vol:
            ti.atomic_sub(vol_acc[None], cell_vol)
        else:
            flags[i, j, k] = FLAG_FLUID
            f_l[i, j, k] = 1.0
            phi[i, j, k] = 1.0
            T[i, j, k] = T_drop
            H[i, j, k] = cp_rho * T_drop + L_rho
            rho[i, j, k] = rho0
            ti.atomic_add(real_acc[None], cell_vol)
            for q in ti.static(range(19)):
                f_src[q, i, j, k] = W[q] * rho0

    # Pass 3: thermal top-up of already-liquid footprint cells. Does NOT consume
    # the deposit budget — previously it filled vol_acc and starved gas→fluid
    # conversion / outer radius retries, driving mass_balance ≪ 1.
    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_lo, k_hi)):
        if flags[i, j, k] != FLAG_FLUID or f_l[i, j, k] < 0.45:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r_ij2 = di * di + dj * dj
        if r_ij2 > foot2 * 1.44:
            continue
        T[i, j, k] = T_drop
        H[i, j, k] = cp_rho * T_drop + L_rho
        rho[i, j, k] = rho0
        for q in ti.static(range(19)):
            f_src[q, i, j, k] = W[q] * rho0


# ──────────────────────────────────────────────────────────────────────────────
@ti.func
def arc_deposition_weight(
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    f_l: ti.f32,
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
) -> ti.f32:
    """
    Gaussian attenuation below the local pool surface (arc_k).

    Liquid pool cells (f_l > 0.55) receive full power. Solid substrate is
    heated with exp(-z/δ) falloff (δ ≈ penetration_cells · Δx), consistent
    with limited arc penetration depth in Rosenthal / Goldak models.
    """
    w = 1.0
    if enable_surface_weight == 1:
        if f_l > 0.55:
            w = 1.0
        else:
            dz = arc_k - ti.f32(k)
            if dz < -0.5:
                w = 0.0
            else:
                depth = ti.max(0.0, dz)
                inv_pen = 1.0 / (penetration_cells + 1e-6)
                w = ti.math.exp(-depth * inv_pen)
    return w

@ti.kernel
def surface_height_at(
    phi: ti.template(),
    flags: ti.template(),
    out: ti.template(),
    i0: ti.i32,
    j0: ti.i32,
    nz_fallback: ti.i32,
    FLAG_GAS: ti.i32,
    nz: ti.i32,
):
    """Top metal cell index at column (i0,j0); falls back to nz_fallback."""
    k_surf = ti.max(1, nz_fallback - 1)
    for k in range(nz):
        if flags[i0, j0, k] != FLAG_GAS and phi[i0, j0, k] > 0.05:
            k_surf = k
    out[None] = ti.f32(k_surf)


@ti.kernel
def feed_wire_momentum(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    f_l: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    droplet_radius: ti.f32,
    vz_lu: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Impart downward droplet momentum to freshly deposited fluid cells."""
    i_min = ti.max(0, ti.cast(arc_i - droplet_radius, ti.i32))
    i_max = ti.min(nx, ti.cast(arc_i + droplet_radius, ti.i32) + 1)
    j_min = ti.max(0, ti.cast(arc_j - droplet_radius, ti.i32))
    j_max = ti.min(ny, ti.cast(arc_j + droplet_radius, ti.i32) + 1)
    k_min = ti.max(0, ti.cast(arc_k, ti.i32))
    k_max = ti.min(nz, ti.cast(arc_k + droplet_radius * 2.0, ti.i32) + 1)

    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
        if flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.5:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        if di * di + dj * dj <= droplet_radius * droplet_radius:
            uz[i, j, k] = ti.min(uz[i, j, k] + vz_lu, 0.15)

@ti.func
def _window_copy_cell(
    i, j, k, si, sj, sk,
    f_a: ti.template(),
    f_b: ti.template(),
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
):
    T[i, j, k] = T[si, sj, sk]
    H[i, j, k] = H[si, sj, sk]
    f_l[i, j, k] = f_l[si, sj, sk]
    phi[i, j, k] = phi[si, sj, sk]
    flags[i, j, k] = flags[si, sj, sk]
    T_max[i, j, k] = T_max[si, sj, sk]
    T_prev[i, j, k] = T_prev[si, sj, sk]
    dT_dt[i, j, k] = dT_dt[si, sj, sk]
    time_above_800[i, j, k] = time_above_800[si, sj, sk]
    time_above_1100[i, j, k] = time_above_1100[si, sj, sk]
    time_above_solidus[i, j, k] = time_above_solidus[si, sj, sk]
    rho[i, j, k] = rho[si, sj, sk]
    ux[i, j, k] = ux[si, sj, sk]
    uy[i, j, k] = uy[si, sj, sk]
    uz[i, j, k] = uz[si, sj, sk]
    Fx[i, j, k] = Fx[si, sj, sk]
    Fy[i, j, k] = Fy[si, sj, sk]
    Fz[i, j, k] = Fz[si, sj, sk]
    cp_rho_field[i, j, k] = cp_rho_field[si, sj, sk]
    alpha_lu_field[i, j, k] = alpha_lu_field[si, sj, sk]
    dgamma_lu_field[i, j, k] = dgamma_lu_field[si, sj, sk]
    tau_field[i, j, k] = tau_field[si, sj, sk]
    alloy_id[i, j, k] = alloy_id[si, sj, sk]
    alloy_frac[i, j, k] = alloy_frac[si, sj, sk]
    for q in ti.static(range(19)):
        f_a[q, i, j, k] = f_a[q, si, sj, sk]
        f_b[q, i, j, k] = f_b[q, si, sj, sk]


@ti.func
def _window_fill_cell(
    i, j, k,
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    f_a: ti.template(),
    f_b: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
    T_amb: ti.f32,
    cp_rho: ti.f32,
    cp_rho_plate: ti.f32,
    rho0: ti.f32,
    as_plate: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    T[i, j, k] = T_amb
    f_l[i, j, k] = 0.0
    rho[i, j, k] = rho0
    ux[i, j, k] = 0.0
    uy[i, j, k] = 0.0
    uz[i, j, k] = 0.0
    Fx[i, j, k] = 0.0
    Fy[i, j, k] = 0.0
    Fz[i, j, k] = 0.0
    T_prev[i, j, k] = T_amb
    dT_dt[i, j, k] = 0.0
    T_max[i, j, k] = T_amb
    time_above_800[i, j, k] = 0.0
    time_above_1100[i, j, k] = 0.0
    time_above_solidus[i, j, k] = 0.0
    if as_plate == 1:
        flags[i, j, k] = FLAG_SOLID
        phi[i, j, k] = 1.0
        alloy_id[i, j, k] = ALLOY_PLATE
        alloy_frac[i, j, k] = 1.0
        H[i, j, k] = cp_rho_plate * T_amb
    else:
        flags[i, j, k] = FLAG_GAS
        phi[i, j, k] = 0.0
        alloy_id[i, j, k] = ALLOY_WIRE
        alloy_frac[i, j, k] = 0.0
        H[i, j, k] = cp_rho * T_amb
    for q in ti.static(range(19)):
        feq = W[q] * rho0
        f_a[q, i, j, k] = feq
        f_b[q, i, j, k] = feq


@ti.kernel
def shift_simulation_window_x(
    n_shift: ti.i32,
    f_a: ti.template(),
    f_b: ti.template(),
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
    T_amb: ti.f32,
    cp_rho: ti.f32,
    cp_rho_plate: ti.f32,
    rho0: ti.f32,
    nz_solid: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Shift all fields left by n_shift cells; reset right strip to ambient substrate/gas.

    The serial inner loop MUST run ascending in i: cell i reads from
    si = i + n_shift (> i), which is only unwritten if lower indices are
    filled first. (A previous descending order re-read already-shifted cells,
    double-shifting ~2/3 of the domain on every window move.)
    """
    for j, k in ti.ndrange(ny, nz):
        for i in range(nx - n_shift):
            si = i + n_shift
            _window_copy_cell(
                i, j, k, si, j, k,
                f_a, f_b, T, H, f_l, phi, flags,
                T_max, T_prev, dT_dt,
                time_above_800, time_above_1100, time_above_solidus,
                rho, ux, uy, uz, Fx, Fy, Fz,
                cp_rho_field, alpha_lu_field, dgamma_lu_field, tau_field,
                alloy_id, alloy_frac,
            )

    i0 = nx - n_shift
    for i, j, k in ti.ndrange(nx, ny, nz):
        if i < i0:
            continue
        as_plate = 1 if k < nz_solid else 0
        _window_fill_cell(
            i, j, k,
            T, H, f_l, phi, flags, T_max, T_prev, dT_dt,
            time_above_800, time_above_1100, time_above_solidus,
            rho, ux, uy, uz, Fx, Fy, Fz, f_a, f_b, alloy_id, alloy_frac,
            T_amb, cp_rho, cp_rho_plate, rho0, as_plate, FLAG_SOLID, FLAG_GAS,
        )


@ti.kernel
def shift_simulation_window_y(
    n_shift: ti.i32,
    direction: ti.i32,
    f_a: ti.template(),
    f_b: ti.template(),
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
    T_amb: ti.f32,
    cp_rho: ti.f32,
    cp_rho_plate: ti.f32,
    rho0: ti.f32,
    nz_solid: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Slide the mesh in ±Y. direction>0: torch +Y (copy down, fill high j)."""
    if direction > 0:
        for i, k in ti.ndrange(nx, nz):
            for j in range(ny - n_shift):
                _window_copy_cell(
                    i, j, k, i, j + n_shift, k,
                    f_a, f_b, T, H, f_l, phi, flags,
                    T_max, T_prev, dT_dt,
                    time_above_800, time_above_1100, time_above_solidus,
                    rho, ux, uy, uz, Fx, Fy, Fz,
                    cp_rho_field, alpha_lu_field, dgamma_lu_field, tau_field,
                    alloy_id, alloy_frac,
                )
        j0 = ny - n_shift
        for i, j, k in ti.ndrange(nx, ny, nz):
            if j < j0:
                continue
            as_plate = 1 if k < nz_solid else 0
            _window_fill_cell(
                i, j, k,
                T, H, f_l, phi, flags, T_max, T_prev, dT_dt,
                time_above_800, time_above_1100, time_above_solidus,
                rho, ux, uy, uz, Fx, Fy, Fz, f_a, f_b, alloy_id, alloy_frac,
                T_amb, cp_rho, cp_rho_plate, rho0, as_plate, FLAG_SOLID, FLAG_GAS,
            )
    else:
        for i, k in ti.ndrange(nx, nz):
            for j in range(ny - n_shift):
                dest = ny - 1 - j
                src = dest - n_shift
                _window_copy_cell(
                    i, dest, k, i, src, k,
                    f_a, f_b, T, H, f_l, phi, flags,
                    T_max, T_prev, dT_dt,
                    time_above_800, time_above_1100, time_above_solidus,
                    rho, ux, uy, uz, Fx, Fy, Fz,
                    cp_rho_field, alpha_lu_field, dgamma_lu_field, tau_field,
                    alloy_id, alloy_frac,
                )
        for i, j, k in ti.ndrange(nx, ny, nz):
            if j >= n_shift:
                continue
            as_plate = 1 if k < nz_solid else 0
            _window_fill_cell(
                i, j, k,
                T, H, f_l, phi, flags, T_max, T_prev, dT_dt,
                time_above_800, time_above_1100, time_above_solidus,
                rho, ux, uy, uz, Fx, Fy, Fz, f_a, f_b, alloy_id, alloy_frac,
                T_amb, cp_rho, cp_rho_plate, rho0, as_plate, FLAG_SOLID, FLAG_GAS,
            )


@ti.kernel
def shift_simulation_window_z(
    n_shift: ti.i32,
    direction: ti.i32,
    f_a: ti.template(),
    f_b: ti.template(),
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    T_max: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    alloy_id: ti.template(),
    alloy_frac: ti.template(),
    T_amb: ti.f32,
    cp_rho: ti.f32,
    cp_rho_plate: ti.f32,
    rho0: ti.f32,
    nz_solid: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Slide the mesh in ±Z. +Z fills the new top with gas; −Z fills the bottom with plate."""
    if direction > 0:
        for i, j in ti.ndrange(nx, ny):
            for k in range(nz - n_shift):
                _window_copy_cell(
                    i, j, k, i, j, k + n_shift,
                    f_a, f_b, T, H, f_l, phi, flags,
                    T_max, T_prev, dT_dt,
                    time_above_800, time_above_1100, time_above_solidus,
                    rho, ux, uy, uz, Fx, Fy, Fz,
                    cp_rho_field, alpha_lu_field, dgamma_lu_field, tau_field,
                    alloy_id, alloy_frac,
                )
        k0 = nz - n_shift
        for i, j, k in ti.ndrange(nx, ny, nz):
            if k < k0:
                continue
            _window_fill_cell(
                i, j, k,
                T, H, f_l, phi, flags, T_max, T_prev, dT_dt,
                time_above_800, time_above_1100, time_above_solidus,
                rho, ux, uy, uz, Fx, Fy, Fz, f_a, f_b, alloy_id, alloy_frac,
                T_amb, cp_rho, cp_rho_plate, rho0, 0, FLAG_SOLID, FLAG_GAS,
            )
    else:
        for i, j in ti.ndrange(nx, ny):
            for k in range(nz - n_shift):
                dest = nz - 1 - k
                src = dest - n_shift
                _window_copy_cell(
                    i, j, dest, i, j, src,
                    f_a, f_b, T, H, f_l, phi, flags,
                    T_max, T_prev, dT_dt,
                    time_above_800, time_above_1100, time_above_solidus,
                    rho, ux, uy, uz, Fx, Fy, Fz,
                    cp_rho_field, alpha_lu_field, dgamma_lu_field, tau_field,
                    alloy_id, alloy_frac,
                )
        for i, j, k in ti.ndrange(nx, ny, nz):
            if k >= n_shift:
                continue
            _window_fill_cell(
                i, j, k,
                T, H, f_l, phi, flags, T_max, T_prev, dT_dt,
                time_above_800, time_above_1100, time_above_solidus,
                rho, ux, uy, uz, Fx, Fy, Fz, f_a, f_b, alloy_id, alloy_frac,
                T_amb, cp_rho, cp_rho_plate, rho0, 1, FLAG_SOLID, FLAG_GAS,
            )


@ti.kernel
def shift_tracers(
    pos: ti.template(),
    active: ti.template(),
    dx_shift: ti.f32,
    dy_shift: ti.f32,
    dz_shift: ti.f32,
    max_tracers: ti.i32,
):
    """Translate tracers after a moving-window field shift (subtract the slide)."""
    for p in range(max_tracers):
        if active[p] != 0:
            pos[p][0] = pos[p][0] - dx_shift
            pos[p][1] = pos[p][1] - dy_shift
            pos[p][2] = pos[p][2] - dz_shift


@ti.kernel
def shift_tracers_x(
    pos: ti.template(),
    active: ti.template(),
    dx_shift: ti.f32,
    max_tracers: ti.i32,
):
    """Translate tracer X by −dx_shift after a +X moving-window field shift."""
    for p in range(max_tracers):
        if active[p] != 0:
            pos[p][0] = pos[p][0] - dx_shift


@ti.kernel
def mix_alloy_fusion_zone(
    alloy_frac: ti.template(),
    alloy_frac_buf: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    mix_rate: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Jacobi average of composition among neighbouring liquid cells.

    Solid / mush cells (f_l < 0.5) freeze their fraction. Gas is skipped.
    Birth ``alloy_id`` is unchanged — this only updates ``alloy_frac``.
    """
    for i, j, k in ti.ndrange(nx, ny, nz):
        alloy_frac_buf[i, j, k] = alloy_frac[i, j, k]
        if flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.5:
            continue
        acc = alloy_frac[i, j, k]
        n = 1.0
        for di, dj, dk in ti.static((
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        )):
            ni = i + di
            nj = j + dj
            nk = k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                if flags[ni, nj, nk] != FLAG_GAS and f_l[ni, nj, nk] >= 0.5:
                    acc += alloy_frac[ni, nj, nk]
                    n += 1.0
        avg = acc / n
        mixed = alloy_frac[i, j, k] * (1.0 - mix_rate) + avg * mix_rate
        alloy_frac_buf[i, j, k] = ti.min(1.0, ti.max(0.0, mixed))
    for i, j, k in ti.ndrange(nx, ny, nz):
        alloy_frac[i, j, k] = alloy_frac_buf[i, j, k]

# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 11 — Porosity Tracer Tracking
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def inject_tracers(
    pos:    ti.template(),
    active: ti.template(),
    head:   ti.template(),   # ti.field(dtype=ti.i32, shape=()) — atomic ring-buffer
    max_tracers: ti.i32,
    arc_x:  ti.f32,
    arc_y:  ti.f32,
    arc_z:  ti.f32,
    sigma_m: ti.f32,
    spawn_count: ti.i32,
):
    """
    Spawn tracer particles using an atomic ring-buffer head pointer.
    O(spawn_count) parallel — no serial scan over all tracers.
    Slots wrap around mod max_tracers, evicting the oldest particles.
    """
    for spawn_idx in range(spawn_count):
        slot = ti.atomic_add(head[None], 1) % max_tracers
        active[slot] = 1
        rdx = (ti.random() * 2.0 - 1.0) * sigma_m
        rdy = (ti.random() * 2.0 - 1.0) * sigma_m
        pos[slot] = ti.Vector([arc_x + rdx, arc_y + rdy, arc_z])

@ti.func
def _sample_trilinear(
    field: ti.template(),
    x: ti.f32,
    y: ti.f32,
    z: ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
) -> ti.f32:
    """Trilinear sample of a cell-centered field. x,y,z in cell-index units."""
    i0 = ti.cast(ti.floor(x), ti.i32)
    j0 = ti.cast(ti.floor(y), ti.i32)
    k0 = ti.cast(ti.floor(z), ti.i32)
    i0 = ti.max(0, ti.min(i0, nx - 1))
    j0 = ti.max(0, ti.min(j0, ny - 1))
    k0 = ti.max(0, ti.min(k0, nz - 1))
    i1 = ti.min(i0 + 1, nx - 1)
    j1 = ti.min(j0 + 1, ny - 1)
    k1 = ti.min(k0 + 1, nz - 1)
    fx = ti.min(1.0, ti.max(0.0, x - ti.cast(i0, ti.f32)))
    fy = ti.min(1.0, ti.max(0.0, y - ti.cast(j0, ti.f32)))
    fz = ti.min(1.0, ti.max(0.0, z - ti.cast(k0, ti.f32)))
    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


@ti.kernel
def advect_tracers(
    pos:    ti.template(),
    active: ti.template(),
    ux:     ti.template(),
    uy:     ti.template(),
    uz:     ti.template(),
    f_l:    ti.template(),
    flags:  ti.template(),
    dx:     ti.f32,
    dt:     ti.f32,
    max_tracers: ti.i32,
    FLAG_SOLID:  ti.i32,
    FLAG_GAS:    ti.i32,
):
    """
    Advect active particles using trilinear-sampled LBM velocity.
    If a particle enters a solidifying cell (f_l < 0.05), it gets trapped.
    If it leaves the domain or enters gas, it becomes inactive.
    """
    nx, ny, nz = ux.shape
    for p in range(max_tracers):
        if active[p] == 1:
            idx = pos[p] / dx
            i = ti.cast(ti.floor(idx.x), ti.i32)
            j = ti.cast(ti.floor(idx.y), ti.i32)
            k = ti.cast(ti.floor(idx.z), ti.i32)

            if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                active[p] = 0
                continue

            flag = flags[i, j, k]
            if flag == FLAG_GAS:
                active[p] = 0
                continue

            if f_l[i, j, k] < 0.05 or flag == FLAG_SOLID:
                active[p] = 2
                continue

            u = ti.Vector([
                _sample_trilinear(ux, idx.x, idx.y, idx.z, nx, ny, nz),
                _sample_trilinear(uy, idx.x, idx.y, idx.z, nx, ny, nz),
                _sample_trilinear(uz, idx.x, idx.y, idx.z, nx, ny, nz),
            ])
            # u is in lattice units (cells per timestep); dx_phys = u * dx
            pos[p] += u * dx


