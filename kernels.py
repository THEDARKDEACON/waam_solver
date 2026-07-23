"""
kernels.py — Pure Taichi GPU Kernels
======================================
ALL simulation physics live here as @ti.kernel functions.
No Python logic runs inside these — they compile to native CUDA PTX.

Physics implemented:
  1. Grid initialisation
  2. Gaussian arc heat injection
  3. Enthalpy-Porosity phase change (f_l update)
  4. Thermal advection-diffusion (upwind scheme)
  5. Free-surface VOF tracking
  6. Marangoni CSF force assembly (Continuum Surface Force)
  7. Boussinesq buoyancy
  8. D3Q19 SRT equilibrium (bootstrap / V&V baseline)
  9. D3Q19 MRT-class collision (production — register-minimized)
 10. Semi-implicit Carman-Kozeny velocity correction
 11. Streaming (pull scheme, bounce-back for solid boundaries)
 12. Macroscopic moment extraction (rho, u)

Register pressure strategy:
  - Intermediate scalars are reused via reassignment rather than
    accumulating named temporaries.
  - All moment transforms are written as in-place sums, not matrix ops.
  - The Cumulant operator is expanded algebraically (offline, via SymPy)
    into the minimal FMA (Fused Multiply-Add) sequence. The full
    symbolic derivation is in tools/derive_cumulant.py.

NOTE: This module must be imported AFTER ti.init() has been called.
"""

import taichi as ti

MAX_KNOTS = 8

# ──────────────────────────────────────────────────────────────────────────────
#  D3Q19 velocity-set constants
# ──────────────────────────────────────────────────────────────────────────────
# Compile-time Python tuples: indexed inside ti.static loops these fold to
# immediates in the generated PTX (no global-memory reads per direction).

EX = ( 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0)
EY = ( 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1)
EZ = ( 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1)
W = (
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
)
OPP = (0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17)


def bind_velocity_set(grid):
    """Kept for API compatibility — velocity set is now compile-time constant."""
    return None


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
    T_ambient: ti.f32,
    rho0:      ti.f32,
    cp_rho:    ti.f32,      # rho * cp  [J/(m³·K)]
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
        H[i, j, k]   = cp_rho * T_ambient   # Initial enthalpy
        # f_l represents liquid fraction. Initially everything is solid (0.0) or gas.
        f_l[i, j, k] = 0.0
        if nz_solid < 0:
            flags[i, j, k] = FLAG_FLUID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 1.0
        else:
            in_plate = (
                i >= plate_i0 and i < plate_i1
                and j >= plate_j0 and j < plate_j1
            )
            if in_plate and k < nz_solid:
                phi[i, j, k] = 1.0
                flags[i, j, k] = FLAG_SOLID
            else:
                phi[i, j, k] = 0.0
                flags[i, j, k] = FLAG_GAS


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
#  Arc deposition weight (surface / penetration attenuation — no RNG)
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


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 2 — Gaussian Arc Heat Injection
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def inject_arc_heat(
    H:     ti.template(),
    flags: ti.template(),
    phi:   ti.template(),
    f_l:   ti.template(),
    norm_buf: ti.template(),  # 0-d scratch: Σ of Gaussian weights over heated cells
    # Arc parameters (all in lattice units)
    arc_i: ti.f32,   # torch x-position [cells]
    arc_j: ti.f32,   # torch y-position [cells]
    arc_k: ti.f32,   # torch z-position (surface level) [cells]
    Q_w:   ti.f32,   # Effective arc power [W]
    sigma: ti.f32,   # Gaussian beam radius [cells]
    dt:    ti.f32,   # Physical timestep [s]
    dx3:   ti.f32,   # Cell volume [m³]
    eta:   ti.f32,   # Arc thermal efficiency [-]
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Distribute arc heat via a Gaussian weight profile, energy-normalized.

    Pass 1 accumulates the total Gaussian×deposition weight over all heated
    cells; pass 2 injects ΔH = η·Q·dt · w_cell / (Σw · dx³). This guarantees
    exactly η·Q·dt joules enter the domain per step regardless of how much of
    the analytic profile is intercepted by metal (the old closed-form 2D
    surface normalization over-injected when the weight extended into depth).
    """
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    norm_buf[None] = 0.0

    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        r2 = di * di + dj * dj + dk * dk * 0.25
        ti.atomic_add(norm_buf[None], ti.math.exp(-r2 * inv2s2) * w_dep)

    # Struct-fors must sit at the kernel's outermost scope (Taichi offload
    # restriction) — guard w_total inside the loop instead of around it.
    w_total = norm_buf[None]
    energy_per_weight = 0.0
    if w_total > 1e-9:
        energy_per_weight = Q_w * eta * dt / (w_total * dx3)
    for i, j, k in H:
        if energy_per_weight <= 0.0 or flags[i, j, k] == FLAG_GAS:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        r2 = di * di + dj * dj + dk * dk * 0.25
        H[i, j, k] += energy_per_weight * ti.math.exp(-r2 * inv2s2) * w_dep


@ti.func
def _goldak_pdf_weight(
    di: ti.f32,
    dj: ti.f32,
    dk: ti.f32,
    travel_sign: ti.f32,
    ff: ti.f32,
    fr: ti.f32,
    a_front: ti.f32,
    a_rear: ti.f32,
    b_axis: ti.f32,
    c_axis: ti.f32,
) -> ti.f32:
    """
    Goldak (1984) spatial factor (relative):

        f_{f,r} exp(-3 x²/a_{f,r}² - 3 y²/b² - 3 z²/c²)

    Absolute 6√3/(π√π a b c) amplitude is absorbed by energy renormalization.
    x is along travel (travel_sign·di ≥ 0 ⇒ front).
    """
    eps = 1e-6
    x_travel = di * travel_sign
    is_front = x_travel >= 0.0
    a_axis = ti.select(is_front, a_front, a_rear)
    frac = ti.select(is_front, ff, fr)
    return frac * ti.math.exp(
        -3.0 * (
            (x_travel * x_travel) / (a_axis * a_axis + eps)
            + (dj * dj) / (b_axis * b_axis + eps)
            + (dk * dk) / (c_axis * c_axis + eps)
        )
    )


@ti.kernel
def inject_goldak_heat(
    H: ti.template(),
    flags: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    norm_buf: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    Q_w: ti.f32,
    dt: ti.f32,
    dx3: ti.f32,
    eta: ti.f32,
    travel_sign: ti.f32,
    ff: ti.f32,
    fr: ti.f32,
    a_front: ti.f32,
    a_rear: ti.f32,
    b_axis: ti.f32,
    c_axis: ti.f32,
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """
    Goldak (1984) double-ellipsoid heat source, energy-normalized.

    Semi-axes a_front / a_rear (travel), b (transverse), c (depth) in cells.
    Requires f_f + f_r = 2 in the loader (relative split still works under
    renormalization). Pass 1 sums weights on metal; pass 2 deposits exactly
    η·Q·dt joules.
    """
    norm_buf[None] = 0.0

    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        w = _goldak_pdf_weight(
            di, dj, dk, travel_sign, ff, fr,
            a_front, a_rear, b_axis, c_axis,
        )
        ti.atomic_add(norm_buf[None], w * w_dep)

    w_total = norm_buf[None]
    energy_per_weight = 0.0
    if w_total > 1e-9:
        energy_per_weight = Q_w * eta * dt / (w_total * dx3)
    for i, j, k in H:
        if energy_per_weight <= 0.0 or flags[i, j, k] == FLAG_GAS:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        w = _goldak_pdf_weight(
            di, dj, dk, travel_sign, ff, fr,
            a_front, a_rear, b_axis, c_axis,
        )
        H[i, j, k] += energy_per_weight * w * w_dep


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
def _solid_wall_normal(
    flags: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    FLAG_SOLID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Unit normal pointing from solid into fluid (sum of outward directions from solid neighbours)."""
    wx = 0.0
    wy = 0.0
    wz = 0.0
    has_wall = 0
    if i > 0 and flags[i - 1, j, k] == FLAG_SOLID:
        wx += 1.0
        has_wall = 1
    if i < nx - 1 and flags[i + 1, j, k] == FLAG_SOLID:
        wx -= 1.0
        has_wall = 1
    if j > 0 and flags[i, j - 1, k] == FLAG_SOLID:
        wy += 1.0
        has_wall = 1
    if j < ny - 1 and flags[i, j + 1, k] == FLAG_SOLID:
        wy -= 1.0
        has_wall = 1
    if k > 0 and flags[i, j, k - 1] == FLAG_SOLID:
        wz += 1.0
        has_wall = 1
    if k < nz - 1 and flags[i, j, k + 1] == FLAG_SOLID:
        wz -= 1.0
        has_wall = 1
    wmag = ti.sqrt(wx * wx + wy * wy + wz * wz)
    if wmag > 1e-8:
        wx /= wmag
        wy /= wmag
        wz /= wmag
    return wx, wy, wz, has_wall


@ti.func
def _correct_normal_contact_angle(
    nx_n: ti.f32,
    ny_n: ti.f32,
    nz_n: ti.f32,
    wx: ti.f32,
    wy: ti.f32,
    wz: ti.f32,
    theta_rad: ti.f32,
):
    """Brackbill-style normal so n̂·n_wall = cos(θ) (θ measured through the liquid)."""
    cos_t = ti.cos(theta_rad)
    sin_t = ti.sin(theta_rad)
    dot = nx_n * wx + ny_n * wy + nz_n * wz
    tx = nx_n - dot * wx
    ty = ny_n - dot * wy
    tz = nz_n - dot * wz
    tmag = ti.sqrt(tx * tx + ty * ty + tz * tz)
    # Taichi requires names defined on all branches before use.
    nx_c = 0.0
    ny_c = 0.0
    nz_c = 0.0
    if tmag > 1e-8:
        nx_c = cos_t * wx + sin_t * tx / tmag
        ny_c = cos_t * wy + sin_t * ty / tmag
        nz_c = cos_t * wz + sin_t * tz / tmag
    else:
        if ti.abs(wz) > 0.5:
            nx_c = sin_t
            ny_c = 0.0
            nz_c = cos_t
        elif ti.abs(wx) > 0.5:
            nx_c = cos_t
            ny_c = sin_t
            nz_c = 0.0
        else:
            nx_c = cos_t * wx + sin_t
            ny_c = cos_t * wy
            nz_c = cos_t * wz
    cmag = ti.sqrt(nx_c * nx_c + ny_c * ny_c + nz_c * nz_c)
    if cmag > 1e-8:
        nx_c /= cmag
        ny_c /= cmag
        nz_c /= cmag
    return nx_c, ny_c, nz_c


@ti.func
def _phi_unit_normal_at(
    phi: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    eps: ti.f32,
):
    """
    Unit interface normal n̂ = ∇φ / |∇φ| and |∇φ| (lattice units).

    Returns (nx, ny, nz, gmag). gmag < eps ⇒ normal is undefined (zeros).
    """
    dpx = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
    dpy = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
    dpz = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
    gmag = ti.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
    nx_n = 0.0
    ny_n = 0.0
    nz_n = 0.0
    if gmag >= eps:
        nx_n = dpx / gmag
        ny_n = dpy / gmag
        nz_n = dpz / gmag
    return nx_n, ny_n, nz_n, gmag


@ti.func
def _brackbill_curvature_at(
    phi: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    eps: ti.f32,
):
    """
    Brackbill CSF curvature κ = -∇·n̂ in lattice units (per cell).

    n̂ is evaluated at neighbour cells so the divergence is of the unit normal,
    not a second difference of φ.
    """
    nx0, ny0, nz0, g0 = _phi_unit_normal_at(phi, i, j, k, nx, ny, nz, eps)
    kappa = 0.0
    # Taichi: no early return inside dynamic if — compute only when |∇φ| is usable.
    if g0 >= eps:
        nxp, _, _, gp = _phi_unit_normal_at(
            phi, ti.min(i + 1, nx - 1), j, k, nx, ny, nz, eps,
        )
        nxm, _, _, gm = _phi_unit_normal_at(
            phi, ti.max(i - 1, 0), j, k, nx, ny, nz, eps,
        )
        _, nyp, _, gp2 = _phi_unit_normal_at(
            phi, i, ti.min(j + 1, ny - 1), k, nx, ny, nz, eps,
        )
        _, nym, _, gm2 = _phi_unit_normal_at(
            phi, i, ti.max(j - 1, 0), k, nx, ny, nz, eps,
        )
        _, _, nzp, gp3 = _phi_unit_normal_at(
            phi, i, j, ti.min(k + 1, nz - 1), nx, ny, nz, eps,
        )
        _, _, nzm, gm3 = _phi_unit_normal_at(
            phi, i, j, ti.max(k - 1, 0), nx, ny, nz, eps,
        )

        # Fall back to centre normal where the neighbour has no interface gradient.
        if gp < eps:
            nxp = nx0
        if gm < eps:
            nxm = nx0
        if gp2 < eps:
            nyp = ny0
        if gm2 < eps:
            nym = ny0
        if gp3 < eps:
            nzp = nz0
        if gm3 < eps:
            nzm = nz0

        div_n = 0.5 * (nxp - nxm) + 0.5 * (nyp - nym) + 0.5 * (nzp - nzm)
        kappa = -div_n
    return kappa, g0


@ti.kernel
def compute_csf_tension(
    phi: ti.template(),
    flags: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    gamma_lu: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    enable_wetting: ti.i32,
    theta_rad: ti.f32,
):
    """
    Brackbill CSF surface tension (additive):

        n̂ = ∇φ / |∇φ|,   κ = -∇·n̂,   F = γ κ ∇φ

    Wetting: when enable_wetting and a solid neighbour exists, replace n̂ with
    the contact-angle-corrected normal (Young / Brackbill wall BC) for the
    force direction. Curvature still uses the φ field (ghost-φ BC should be
    applied before this kernel). No empirical sinθ lateral drive.
    """
    eps = 1e-6
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        # Full ±1 stencil for neighbour normals; skip domain rim.
        if (
            i < 1 or j < 1 or k < 1
            or i > nx - 2 or j > ny - 2 or k > nz - 2
        ):
            continue

        kappa, gmag = _brackbill_curvature_at(phi, i, j, k, nx, ny, nz, eps)
        if gmag < eps:
            continue

        dpx = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dpy = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dpz = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])

        fx = gamma_lu * kappa * dpx
        fy = gamma_lu * kappa * dpy
        fz = gamma_lu * kappa * dpz

        if enable_wetting != 0:
            wx, wy, wz, has_wall = _solid_wall_normal(
                flags, i, j, k, FLAG_SOLID, nx, ny, nz,
            )
            if has_wall != 0:
                nx_n = dpx / gmag
                ny_n = dpy / gmag
                nz_n = dpz / gmag
                nx_c, ny_c, nz_c = _correct_normal_contact_angle(
                    nx_n, ny_n, nz_n, wx, wy, wz, theta_rad,
                )
                # F = γ κ n̂_corr |∇φ|  (same magnitude, Young-consistent direction)
                fx = gamma_lu * kappa * nx_c * gmag
                fy = gamma_lu * kappa * ny_c * gmag
                fz = gamma_lu * kappa * nz_c * gmag

        Fx[i, j, k] += fx
        Fy[i, j, k] += fy
        Fz[i, j, k] += fz


# Alias for BEAD_GEOMETRY_PHYSICS_SPEC (wall CSF is integrated in compute_csf_tension).
compute_csf_wetting = compute_csf_tension


@ti.kernel
def apply_contact_angle_phi_bc(
    phi: ti.template(),
    flags: ti.template(),
    theta_rad: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Ghost-fluid φ BC enforcing static contact angle θ at solid walls (Ding & Spelt style).

    Sets gas-cell φ adjacent to the triple line so ∇φ is consistent with θ before CSF.
    """
    cos_t = ti.cos(theta_rad)
    sin_t = ti.sin(theta_rad)
    film = 0.5 * (1.0 - cos_t)
    for i, j, k in phi:
        if flags[i, j, k] == FLAG_SOLID:
            continue
        # Gas above bare substrate — precursor film thickness ∝ (1 - cos θ)
        if k > 0 and flags[i, j, k - 1] == FLAG_SOLID and flags[i, j, k] == FLAG_GAS:
            if phi[i, j, k] < film:
                phi[i, j, k] = film
        # Fluid on substrate: ghost gas above enforces interface slope at θ
        if k > 0 and flags[i, j, k - 1] == FLAG_SOLID and phi[i, j, k] > 0.55:
            if k + 1 < nz and flags[i, j, k + 1] == FLAG_GAS:
                phi_ghost = phi[i, j, k] - sin_t
                phi_ghost = ti.max(film, ti.min(phi_ghost, 0.98))
                if phi[i, j, k + 1] < phi_ghost:
                    phi[i, j, k + 1] = phi_ghost
        # Gas directly above pool fluid (vertical interface segment)
        if k > 0 and phi[i, j, k - 1] > 0.55 and flags[i, j, k] == FLAG_GAS:
            phi_target = phi[i, j, k - 1] - sin_t
            phi_target = ti.max(film, ti.min(phi_target, 0.95))
            if phi[i, j, k] < phi_target:
                phi[i, j, k] = phi_target
        # Horizontal spread from pool at same k (lateral wetting toe)
        if phi[i, j, k] < 0.45:
            for dj in ti.static([-1, 1]):
                jj = j + dj
                if jj >= 0 and jj < ny:
                    if phi[i, jj, k] > 0.55:
                        spread = film * (0.85 + 0.15 * sin_t)
                        if phi[i, j, k] < spread:
                            phi[i, j, k] = spread
            for di in ti.static([-1, 1]):
                ii = i + di
                if ii >= 0 and ii < nx:
                    if phi[ii, j, k] > 0.55:
                        spread = film * (0.85 + 0.15 * sin_t)
                        if phi[i, j, k] < spread:
                            phi[i, j, k] = spread
        # Fluid on solid: pin full liquid at contact
        if k > 0 and flags[i, j, k - 1] == FLAG_SOLID and phi[i, j, k] > 0.2:
            if phi[i, j, k] < 0.65:
                phi[i, j, k] = 0.65


@ti.kernel
def apply_vapor_recoil(
    Fz: ti.template(),
    T: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma: ti.f32,
    recoil_pa: ti.f32,
    T_ref: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Vapor recoil pressure on the free surface (downward, T-dependent)."""
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)
    F_peak_lu = (recoil_pa / (rho_ref * dx)) * dt * dt / dx

    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        dphi_z = phi[i, j, ti.min(k + 1, Fz.shape[2] - 1)] - phi[i, j, ti.max(k - 1, 0)]
        if ti.abs(dphi_z) < 0.05:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r2 = di * di + dj * dj
        t_ratio = ti.min(T[i, j, k] / (T_ref + eps), 3.0)
        force = -F_peak_lu * t_ratio * ti.math.exp(-r2 * inv2s2)
        Fz[i, j, k] += force


# ── Advanced weld forces (recoil CC, gas shear, Lorentz, droplet) ────────────
@ti.func
def _wf_pressure_to_Fz_lu(pressure_pa, dt, dx, rho_ref):
    F_peak_phys = pressure_pa / (rho_ref * dx)
    return F_peak_phys * dt * dt / dx


@ti.func
def _sigma_at(sigma, i, j, k, nx, ny, nz):
    ii = ti.max(0, ti.min(nx - 1, i))
    jj = ti.max(0, ti.min(ny - 1, j))
    kk = ti.max(0, ti.min(nz - 1, k))
    return sigma[ii, jj, kk]


@ti.kernel
def apply_vapor_recoil_clausius_clapeyron(
    Fz: ti.template(),
    T: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma: ti.f32,
    P_ref_Pa: ti.f32,
    T_boil_K: ti.f32,
    L_vapor_J_kg: ti.f32,
    R_spec_J_kgK: ti.f32,
    C_acc: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """
    Vapor recoil via Clausius–Clapeyron: p = C_acc · P_sat(T).

    C_acc ≈ 0.54 (Anisimov / Knight accommodation). Force is zero for T ≤ T_boil.
    """
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)
    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        dphi_z = phi[i, j, ti.min(k + 1, Fz.shape[2] - 1)] - phi[i, j, ti.max(k - 1, 0)]
        if ti.abs(dphi_z) < 0.05:
            continue
        Tc = T[i, j, k]
        if Tc <= T_boil_K:
            continue
        exponent = (L_vapor_J_kg / (R_spec_J_kgK + eps)) * (1.0 / (T_boil_K + eps) - 1.0 / (Tc + eps))
        exponent = ti.min(exponent, 12.0)
        P_vap = C_acc * P_ref_Pa * ti.math.exp(exponent)
        F_peak_lu = _wf_pressure_to_Fz_lu(P_vap, dt, dx, rho_ref)
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r2 = di * di + dj * dj
        Fz[i, j, k] += -F_peak_lu * ti.math.exp(-r2 * inv2s2)


@ti.kernel
def apply_gas_shear_stress(
    Fx: ti.template(),
    Fy: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma: ti.f32,
    tau_peak_pa: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)
    F_scale = _wf_pressure_to_Fz_lu(1.0, dt, dx, rho_ref)
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        dpx = 0.5 * (phi[ti.min(i + 1, Fx.shape[0] - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dpy = 0.5 * (phi[i, ti.min(j + 1, Fx.shape[1] - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dpz = 0.5 * (phi[i, j, ti.min(k + 1, Fx.shape[2] - 1)] - phi[i, j, ti.max(k - 1, 0)])
        gmag = ti.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
        if gmag < 0.08:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r2 = di * di + dj * dj
        r = ti.sqrt(r2 + eps)
        rx = di / r
        ry = dj / r
        nx_n = dpx / gmag
        ny_n = dpy / gmag
        rdotn = rx * nx_n + ry * ny_n
        tx = rx - rdotn * nx_n
        ty = ry - rdotn * ny_n
        tmag = ti.sqrt(tx * tx + ty * ty + eps)
        tx /= tmag
        ty /= tmag
        tau = tau_peak_pa * ti.math.exp(-r2 * inv2s2)
        Fmag = tau * F_scale
        Fx[i, j, k] += Fmag * tx
        Fy[i, j, k] += Fmag * ty


@ti.kernel
def apply_droplet_impact_pressure(
    Fz: ti.template(),
    flags: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    drop_radius: ti.f32,
    impact_pa: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * drop_radius * drop_radius + eps)
    F_peak_lu = _wf_pressure_to_Fz_lu(impact_pa, dt, dx, rho_ref)
    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.25 and phi[i, j, k] < 0.25:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        r2 = di * di + dj * dj + dk * dk
        if r2 > drop_radius * drop_radius * 4.0:
            continue
        Fz[i, j, k] += -F_peak_lu * ti.math.exp(-r2 * inv2s2)


@ti.kernel
def feed_wire_momentum_impact(
    f_src: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    f_l: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    droplet_radius: ti.f32,
    vx_lu: ti.f32,
    vy_lu: ti.f32,
    vz_lu: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Impose droplet impact velocity on the LBM state itself.

    Writing only ux/uy/uz would be overwritten by the next moment extraction;
    the momentum must live in the distributions. Cells in the impact region
    are reset to the equilibrium at (rho_local, v_drop).
    """
    i_min = ti.max(0, ti.cast(arc_i - droplet_radius * 2.5, ti.i32))
    i_max = ti.min(nx, ti.cast(arc_i + droplet_radius * 2.5, ti.i32) + 1)
    j_min = ti.max(0, ti.cast(arc_j - droplet_radius * 2.5, ti.i32))
    j_max = ti.min(ny, ti.cast(arc_j + droplet_radius * 2.5, ti.i32) + 1)
    k_min = ti.max(0, ti.cast(arc_k, ti.i32))
    k_max = ti.min(nz, ti.cast(arc_k + droplet_radius * 3.0, ti.i32) + 1)
    drop_cz = arc_k + droplet_radius + 1.0
    u2 = vx_lu * vx_lu + vy_lu * vy_lu + vz_lu * vz_lu
    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
        if flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.45:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - drop_cz
        r2 = di * di + dj * dj + dk * dk
        r_ij2 = di * di + dj * dj
        in_drop = r2 <= droplet_radius * droplet_radius or (
            r_ij2 <= droplet_radius * droplet_radius and ti.f32(k) >= arc_k + 1.0
        )
        if in_drop:
            r_loc = rho[i, j, k]
            ux[i, j, k] = vx_lu
            uy[i, j, k] = vy_lu
            uz[i, j, k] = vz_lu
            for q in ti.static(range(19)):
                eu = EX[q] * vx_lu + EY[q] * vy_lu + EZ[q] * vz_lu
                f_src[q, i, j, k] = W[q] * r_loc * (
                    1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
                )


@ti.kernel
def elec_build_sigma(
    sigma: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    sigma_liquid: ti.f32,
    sigma_solid: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in sigma:
        if flags[i, j, k] == FLAG_GAS:
            sigma[i, j, k] = 1e-8
        elif f_l[i, j, k] > 0.55:
            sigma[i, j, k] = sigma_liquid
        else:
            sigma[i, j, k] = sigma_solid


@ti.kernel
def elec_clear_source(source: ti.template()):
    for I in ti.grouped(source):
        source[I] = 0.0


@ti.kernel
def elec_inject_arc_source(
    source: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma_cells: ti.f32,
    current_A: ti.f32,
    dx: ti.f32,
    nz: ti.i32,
):
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma_cells * sigma_cells + eps)
    cell_vol = dx * dx * dx
    k_max = ti.min(nz, ti.cast(arc_k + sigma_cells * 4.0, ti.i32) + 1)
    k_min = ti.max(0, ti.cast(arc_k - sigma_cells, ti.i32))
    for i, j, k in source:
        if k < k_min or k > k_max:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        r2 = di * di + dj * dj + 0.25 * dk * dk
        weight = ti.math.exp(-r2 * inv2s2)
        source[i, j, k] = current_A * weight / (cell_vol + eps)


@ti.kernel
def elec_normalize_source(
    source: ti.template(),
    target_current_A: ti.f32,
    dx: ti.f32,
):
    """Scale volume source [A/m³] so ∫ source dV = target_current_A."""
    cell_vol = dx * dx * dx
    total = 0.0
    for i, j, k in source:
        total += source[i, j, k] * cell_vol
    scale = target_current_A / ti.max(total, 1e-12)
    for i, j, k in source:
        source[i, j, k] *= scale


@ti.kernel
def elec_init_ground(
    phi: ti.template(),
    flags: ti.template(),
    nz_solid: ti.i32,
    FLAG_SOLID: ti.i32,
):
    for i, j, k in phi:
        phi[i, j, k] = 0.0


@ti.kernel
def elec_jacobi_step(
    phi_in: ti.template(),
    phi_out: ti.template(),
    sigma: ti.template(),
    source: ti.template(),
    flags: ti.template(),
    dx: ti.f32,
    omega_j: ti.f32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    nz_solid: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    eps = 1e-12
    for i, j, k in phi_out:
        if flags[i, j, k] == FLAG_GAS:
            phi_out[i, j, k] = 0.0
            continue
        if k <= nz_solid and flags[i, j, k] == FLAG_SOLID:
            phi_out[i, j, k] = 0.0
            continue
        sp = _sigma_at(sigma, i + 1, j, k, nx, ny, nz)
        sm = _sigma_at(sigma, i - 1, j, k, nx, ny, nz)
        sn = _sigma_at(sigma, i, j + 1, k, nx, ny, nz)
        ss = _sigma_at(sigma, i, j - 1, k, nx, ny, nz)
        st = _sigma_at(sigma, i, j, k + 1, nx, ny, nz)
        sb = _sigma_at(sigma, i, j, k - 1, nx, ny, nz)
        denom = sp + sm + sn + ss + st + sb + eps
        phi_neighbors = (
            sp * phi_in[ti.min(i + 1, nx - 1), j, k]
            + sm * phi_in[ti.max(i - 1, 0), j, k]
            + sn * phi_in[i, ti.min(j + 1, ny - 1), k]
            + ss * phi_in[i, ti.max(j - 1, 0), k]
            + st * phi_in[i, j, ti.min(k + 1, nz - 1)]
            + sb * phi_in[i, j, ti.max(k - 1, 0)]
        )
        phi_new = (phi_neighbors + source[i, j, k] * dx * dx) / denom
        phi_out[i, j, k] = (1.0 - omega_j) * phi_in[i, j, k] + omega_j * phi_new


@ti.kernel
def elec_compute_J(
    Jx: ti.template(),
    Jy: ti.template(),
    Jz: ti.template(),
    phi: ti.template(),
    sigma: ti.template(),
    flags: ti.template(),
    dx: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    for i, j, k in Jx:
        if flags[i, j, k] == FLAG_GAS:
            Jx[i, j, k] = 0.0
            Jy[i, j, k] = 0.0
            Jz[i, j, k] = 0.0
            continue
        sig = sigma[i, j, k]
        dphidx = (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k]) / (2.0 * dx)
        dphidy = (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k]) / (2.0 * dx)
        dphidz = (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)]) / (2.0 * dx)
        Jx[i, j, k] = -sig * dphidx
        Jy[i, j, k] = -sig * dphidy
        Jz[i, j, k] = -sig * dphidz


@ti.kernel
def elec_l1_diff(
    a: ti.template(),
    b: ti.template(),
    diff_buf: ti.template(),
    norm_buf: ti.template(),
):
    """L1 norm of (a - b) and of b — for Jacobi convergence monitoring."""
    diff_buf[None] = 0.0
    norm_buf[None] = 0.0
    for I in ti.grouped(a):
        ti.atomic_add(diff_buf[None], ti.abs(a[I] - b[I]))
        ti.atomic_add(norm_buf[None], ti.abs(b[I]))


@ti.kernel
def elec_bin_axial_current(
    Jz: ti.template(),
    bins: ti.template(),   # ti.field shape (nz, n_bins)
    arc_i: ti.f32,
    arc_j: ti.f32,
    dx: ti.f32,
    bin_dr_cells: ti.f32,
    n_bins: ti.i32,
    nz: ti.i32,
):
    """Radial histogram of axial current Jz·dA per z-slab around the arc axis."""
    for k, b in ti.ndrange(nz, n_bins):
        bins[k, b] = 0.0
    for i, j, k in Jz:
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r = ti.sqrt(di * di + dj * dj)
        b = ti.min(ti.cast(r / bin_dr_cells, ti.i32), n_bins - 1)
        ti.atomic_add(bins[k, b], Jz[i, j, k] * dx * dx)


@ti.kernel
def elec_prefix_bins(
    bins: ti.template(),
    n_bins: ti.i32,
    nz: ti.i32,
):
    """In-place radial prefix sum: bins[k, b] ← I_enc(r ≤ (b+1)·Δr, k)."""
    for k in range(nz):
        acc = 0.0
        for b in range(n_bins):
            acc += bins[k, b]
            bins[k, b] = acc


@ti.kernel
def elec_B_axisymmetric(
    Bx: ti.template(),
    By: ti.template(),
    Bz: ti.template(),
    bins: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    dx: ti.f32,
    bin_dr_cells: ti.f32,
    mu0: ti.f32,
    n_bins: ti.i32,
    nz: ti.i32,
):
    """
    Self-magnetic field of the welding current (axisymmetric Ampère law):

        B_θ(r, z) = μ0 · I_enc(r, z) / (2π r)

    with I_enc the axial current enclosed within radius r at height z.
    This is the standard GTAW/GMAW electromagnetic pool model (Kou) and
    replaces the previous local μ0·∇×J estimate, which is not a solution
    of ∇×B = μ0·J and had the wrong magnitude and structure.
    """
    for i, j, k in Bx:
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        r_cells = ti.sqrt(di * di + dj * dj)
        Bx[i, j, k] = 0.0
        By[i, j, k] = 0.0
        Bz[i, j, k] = 0.0
        if r_cells > 1e-3:
            b = ti.min(ti.cast(r_cells / bin_dr_cells, ti.i32), n_bins - 1)
            I_enc = bins[k, b]
            r_m = r_cells * dx
            B_theta = mu0 * I_enc / (2.0 * ti.math.pi * r_m)
            # θ̂ = ẑ × r̂ = (−dj, di, 0)/r
            Bx[i, j, k] = -B_theta * dj / r_cells
            By[i, j, k] = B_theta * di / r_cells


@ti.kernel
def apply_lorentz_JxB(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    Jx: ti.template(),
    Jy: ti.template(),
    Jz: ti.template(),
    Bx: ti.template(),
    By: ti.template(),
    Bz: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_GAS or f_l[i, j, k] < 0.05:
            continue
        jx = Jx[i, j, k]
        jy = Jy[i, j, k]
        jz = Jz[i, j, k]
        bx = Bx[i, j, k]
        by = By[i, j, k]
        bz = Bz[i, j, k]
        # J [A/m²] × B [T] → N/m³; divide by ρ [kg/m³] → m/s²; Guo: a_lu = a_phys·dt²/dx
        fx_phys = jy * bz - jz * by
        fy_phys = jz * bx - jx * bz
        fz_phys = jx * by - jy * bx
        fl_w = ti.min(1.0, f_l[i, j, k])
        scale = fl_w * dt * dt / (rho_ref * dx)
        Fx[i, j, k] += fx_phys * scale
        Fy[i, j, k] += fy_phys * scale
        Fz[i, j, k] += fz_phys * scale


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 3 — Enthalpy-Porosity Phase Change
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def update_phase(
    H:   ti.template(),
    T:   ti.template(),
    f_l: ti.template(),
    cp_rho:    ti.f32,   # ρ·cp  [J/(m³·K)]
    L_rho:     ti.f32,   # ρ·L   [J/m³]  latent heat per unit volume
    T_solidus: ti.f32,   # [K]
    T_liquidus: ti.f32,  # [K]
):
    """
    Recover temperature and liquid fraction from the enthalpy field.

    The Enthalpy-Porosity method treats enthalpy H as the primary
    conserved variable. T and f_l are derived quantities.

    Mushy zone: H_sol < H < H_liq
      f_l = (H - H_sol) / (H_liq - H_sol)
      T   = T_solidus + f_l * (T_liquidus - T_solidus)

    Solid:  H ≤ H_sol  →  f_l = 0,  T = H / (ρ·cp)
    Liquid: H ≥ H_liq  →  f_l = 1,  T = T_liq + (H - H_liq) / (ρ·cp)
    """
    H_sol = cp_rho * T_solidus
    H_liq = H_sol + L_rho

    for i, j, k in H:
        h = H[i, j, k]
        if h <= H_sol:
            f_l[i, j, k] = 0.0
            T[i, j, k]   = h / cp_rho
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
            T[i, j, k]   = T_liquidus + (h - H_liq) / cp_rho
        else:
            fl = (h - H_sol) / L_rho
            f_l[i, j, k] = fl
            T[i, j, k]   = T_solidus + fl * (T_liquidus - T_solidus)


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 4 — Thermal Advection-Diffusion
#  Enthalpy advection (latent heat is carried by the flow), central-difference
#  diffusion, minmod-limited 2nd-order upwind, solid conduction, adiabatic
#  gas faces (radiation/convection losses are handled separately).
# ──────────────────────────────────────────────────────────────────────────────
@ti.func
def _minmod(a: ti.f32, b: ti.f32) -> ti.f32:
    """minmod limiter: smaller-magnitude argument when signs agree, else 0."""
    out = 0.0
    if a * b > 0.0:
        out = ti.select(ti.abs(a) < ti.abs(b), a, b)
    return out


@ti.func
def _masked_at(
    fld: ti.template(),
    flags: ti.template(),
    i: ti.i32, j: ti.i32, k: ti.i32,
    fallback: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
) -> ti.f32:
    """Field value at clamped (i,j,k); gas cells return `fallback` (adiabatic)."""
    ii = ti.max(0, ti.min(i, nx - 1))
    jj = ti.max(0, ti.min(j, ny - 1))
    kk = ti.max(0, ti.min(k, nz - 1))
    val = fld[ii, jj, kk]
    if flags[ii, jj, kk] == FLAG_GAS:
        val = fallback
    return val


@ti.func
def _limited_upwind_grad(
    u: ti.f32,
    q_mm: ti.f32, q_m: ti.f32, q_c: ti.f32, q_p: ti.f32, q_pp: ti.f32,
) -> ti.f32:
    """Minmod-limited 2nd-order upwind derivative of q along one axis (Δ=1)."""
    g = 0.0
    if u > 0.0:
        d1 = q_c - q_m
        d2 = 0.5 * (3.0 * q_c - 4.0 * q_m + q_mm)
        g = _minmod(d1, d2)
    else:
        d1 = q_p - q_c
        d2 = 0.5 * (-3.0 * q_c + 4.0 * q_p - q_pp)
        g = _minmod(d1, d2)
    return g


@ti.kernel
def advect_diffuse_temperature(
    H:    ti.template(),    # Updated in-place
    T:    ti.template(),
    ux:   ti.template(),
    uy:   ti.template(),
    uz:   ti.template(),
    flags: ti.template(),
    alpha_lu: ti.f32,       # Thermal diffusivity [lu²/ts]
    dt:       ti.f32,       # Timestep  [ts] (=1 in LBM, but track for physics)
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    cp_rho:     ti.f32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Solve ∂H/∂t + u·∇H = ∇·(k∇T) in lattice units.

    - Advection acts on H (the conserved variable): latent heat is transported
      with the melt, not just sensible heat.
    - SOLID cells conduct (no advection) — the substrate/frozen bead is a real
      heat sink instead of an insulator.
    - GAS neighbours are adiabatic: previously they acted as fixed ambient-T
      conductors, an unphysical heat sink at the free surface.
    - Advection is minmod-limited 2nd-order upwind (less numerical smearing of
      the fusion front than the previous 1st-order scheme).
    """
    for i, j, k in H:
        flag = flags[i, j, k]
        if flag == FLAG_GAS:
            continue

        T_c = T[i, j, k]
        H_c = H[i, j, k]

        # --- Diffusion: central differences on T, gas faces adiabatic ---
        T_im = _masked_at(T, flags, i - 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_ip = _masked_at(T, flags, i + 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jm = _masked_at(T, flags, i, j - 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jp = _masked_at(T, flags, i, j + 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_km = _masked_at(T, flags, i, j, k - 1, T_c, FLAG_GAS, nx, ny, nz)
        T_kp = _masked_at(T, flags, i, j, k + 1, T_c, FLAG_GAS, nx, ny, nz)
        lap = T_ip + T_im + T_jp + T_jm + T_kp + T_km - 6.0 * T_c

        dH = cp_rho * alpha_lu * lap * dt

        if flag != FLAG_SOLID:
            u = ux[i, j, k]
            v = uy[i, j, k]
            w = uz[i, j, k]

            H_im = _masked_at(H, flags, i - 1, j, k, H_c, FLAG_GAS, nx, ny, nz)
            H_ip = _masked_at(H, flags, i + 1, j, k, H_c, FLAG_GAS, nx, ny, nz)
            H_imm = _masked_at(H, flags, i - 2, j, k, H_im, FLAG_GAS, nx, ny, nz)
            H_ipp = _masked_at(H, flags, i + 2, j, k, H_ip, FLAG_GAS, nx, ny, nz)
            H_jm = _masked_at(H, flags, i, j - 1, k, H_c, FLAG_GAS, nx, ny, nz)
            H_jp = _masked_at(H, flags, i, j + 1, k, H_c, FLAG_GAS, nx, ny, nz)
            H_jmm = _masked_at(H, flags, i, j - 2, k, H_jm, FLAG_GAS, nx, ny, nz)
            H_jpp = _masked_at(H, flags, i, j + 2, k, H_jp, FLAG_GAS, nx, ny, nz)
            H_km = _masked_at(H, flags, i, j, k - 1, H_c, FLAG_GAS, nx, ny, nz)
            H_kp = _masked_at(H, flags, i, j, k + 1, H_c, FLAG_GAS, nx, ny, nz)
            H_kmm = _masked_at(H, flags, i, j, k - 2, H_km, FLAG_GAS, nx, ny, nz)
            H_kpp = _masked_at(H, flags, i, j, k + 2, H_kp, FLAG_GAS, nx, ny, nz)

            gx = _limited_upwind_grad(u, H_imm, H_im, H_c, H_ip, H_ipp)
            gy = _limited_upwind_grad(v, H_jmm, H_jm, H_c, H_jp, H_jpp)
            gz = _limited_upwind_grad(w, H_kmm, H_km, H_c, H_kp, H_kpp)
            dH -= (u * gx + v * gy + w * gz) * dt

        H[i, j, k] += dH


@ti.func
def lookup_table_1d(
    T: ti.f32,
    t_arr: ti.template(),
    v_arr: ti.template(),
    n: ti.i32,
    fallback: ti.f32,
) -> ti.f32:
    result = fallback
    if n > 0:
        if T <= t_arr[0]:
            result = v_arr[0]
        elif T >= t_arr[n - 1]:
            result = v_arr[n - 1]
        else:
            for idx in ti.static(range(MAX_KNOTS - 1)):
                if idx < n - 1:
                    t0 = t_arr[idx]
                    t1 = t_arr[idx + 1]
                    if T >= t0 and T <= t1:
                        w = (T - t0) / (t1 - t0 + 1e-8)
                        result = v_arr[idx] + w * (v_arr[idx + 1] - v_arr[idx])
    return result


@ti.kernel
def refresh_thermal_properties(
    T: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    cp_T: ti.template(),
    cp_V: ti.template(),
    k_T: ti.template(),
    k_V: ti.template(),
    mu_T: ti.template(),
    mu_V: ti.template(),
    dgamma_T: ti.template(),
    dgamma_V: ti.template(),
    n_cp: ti.template(),
    n_k: ti.template(),
    n_mu: ti.template(),
    n_dgamma: ti.template(),
    rho: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    cp_fallback: ti.f32,
    k_fallback: ti.f32,
    mu_fallback: ti.f32,
    dgamma_fallback: ti.f32,
    force_scale: ti.f32,
    cp_rho_ref: ti.f32,
    alpha_lu_ref: ti.f32,
    dgamma_lu_ref: ti.f32,
    tau_ref: ti.f32,
    marangoni_scale: ti.f32,
    use_tables: ti.i32,
    flags: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in T:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            cp_rho_field[i, j, k] = cp_rho_ref
            alpha_lu_field[i, j, k] = alpha_lu_ref
            dgamma_lu_field[i, j, k] = dgamma_lu_ref
            tau_field[i, j, k] = tau_ref
            continue
        T_c = T[i, j, k]
        if use_tables == 1:
            cp = lookup_table_1d(T_c, cp_T, cp_V, n_cp[None], cp_fallback)
            k_val = lookup_table_1d(T_c, k_T, k_V, n_k[None], k_fallback)
            mu_val = lookup_table_1d(T_c, mu_T, mu_V, n_mu[None], mu_fallback)
            dgamma = lookup_table_1d(T_c, dgamma_T, dgamma_V, n_dgamma[None], dgamma_fallback)
            cp_r = rho * cp
            alpha_phys = k_val / (rho * cp + 1e-9)
            nu_phys = mu_val / (rho + 1e-9)
            nu_lu = nu_phys * dt / (dx * dx)
            cp_rho_field[i, j, k] = cp_r
            alpha_lu_field[i, j, k] = alpha_phys * dt / (dx * dx)
            dgamma_lu_field[i, j, k] = dgamma * force_scale * marangoni_scale
            tau_field[i, j, k] = 3.0 * nu_lu + 0.5
        else:
            cp_rho_field[i, j, k] = cp_rho_ref
            alpha_lu_field[i, j, k] = alpha_lu_ref
            dgamma_lu_field[i, j, k] = dgamma_lu_ref * marangoni_scale
            tau_field[i, j, k] = tau_ref


@ti.kernel
def advect_diffuse_temperature_variable(
    H: ti.template(),
    T: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    alpha_lu_field: ti.template(),
    cp_rho_field: ti.template(),
    dt: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Variable-property version of advect_diffuse_temperature (same scheme)."""
    for i, j, k in H:
        flag = flags[i, j, k]
        if flag == FLAG_GAS:
            continue

        T_c = T[i, j, k]
        H_c = H[i, j, k]
        alpha_lu = alpha_lu_field[i, j, k]
        cp_rho = cp_rho_field[i, j, k]

        T_im = _masked_at(T, flags, i - 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_ip = _masked_at(T, flags, i + 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jm = _masked_at(T, flags, i, j - 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jp = _masked_at(T, flags, i, j + 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_km = _masked_at(T, flags, i, j, k - 1, T_c, FLAG_GAS, nx, ny, nz)
        T_kp = _masked_at(T, flags, i, j, k + 1, T_c, FLAG_GAS, nx, ny, nz)
        lap = T_ip + T_im + T_jp + T_jm + T_kp + T_km - 6.0 * T_c

        dH = cp_rho * alpha_lu * lap * dt

        if flag != FLAG_SOLID:
            u = ux[i, j, k]
            v = uy[i, j, k]
            w = uz[i, j, k]

            H_im = _masked_at(H, flags, i - 1, j, k, H_c, FLAG_GAS, nx, ny, nz)
            H_ip = _masked_at(H, flags, i + 1, j, k, H_c, FLAG_GAS, nx, ny, nz)
            H_imm = _masked_at(H, flags, i - 2, j, k, H_im, FLAG_GAS, nx, ny, nz)
            H_ipp = _masked_at(H, flags, i + 2, j, k, H_ip, FLAG_GAS, nx, ny, nz)
            H_jm = _masked_at(H, flags, i, j - 1, k, H_c, FLAG_GAS, nx, ny, nz)
            H_jp = _masked_at(H, flags, i, j + 1, k, H_c, FLAG_GAS, nx, ny, nz)
            H_jmm = _masked_at(H, flags, i, j - 2, k, H_jm, FLAG_GAS, nx, ny, nz)
            H_jpp = _masked_at(H, flags, i, j + 2, k, H_jp, FLAG_GAS, nx, ny, nz)
            H_km = _masked_at(H, flags, i, j, k - 1, H_c, FLAG_GAS, nx, ny, nz)
            H_kp = _masked_at(H, flags, i, j, k + 1, H_c, FLAG_GAS, nx, ny, nz)
            H_kmm = _masked_at(H, flags, i, j, k - 2, H_km, FLAG_GAS, nx, ny, nz)
            H_kpp = _masked_at(H, flags, i, j, k + 2, H_kp, FLAG_GAS, nx, ny, nz)

            gx = _limited_upwind_grad(u, H_imm, H_im, H_c, H_ip, H_ipp)
            gy = _limited_upwind_grad(v, H_jmm, H_jm, H_c, H_jp, H_jpp)
            gz = _limited_upwind_grad(w, H_kmm, H_km, H_c, H_kp, H_kpp)
            dH -= (u * gx + v * gy + w * gz) * dt

        H[i, j, k] += dH


@ti.kernel
def update_phase_variable_cp(
    H: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    cp_rho_field: ti.template(),
    L_rho: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    H_sol: ti.f32,
    H_liq: ti.f32,
):
    for i, j, k in H:
        h = H[i, j, k]
        cp_rho = cp_rho_field[i, j, k]
        if h <= H_sol:
            f_l[i, j, k] = 0.0
            T[i, j, k] = h / (cp_rho + 1e-9)
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
            T[i, j, k] = T_liquidus + (h - H_liq) / (cp_rho + 1e-9)
        else:
            fl = (h - H_sol) / L_rho
            f_l[i, j, k] = fl
            T[i, j, k] = T_solidus + fl * (T_liquidus - T_solidus)


@ti.kernel
def apply_thermal_boundary_losses_variable(
    H: ti.template(),
    T: ti.template(),
    flags: ti.template(),
    cp_rho_field: ti.template(),
    T_amb: ti.f32,
    h_conv: ti.f32,
    eps_rad: ti.f32,
    enable_conv: ti.i32,
    enable_rad: ti.i32,
    dt: ti.f32,
    dx: ti.f32,
    sigma_sb: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Convective/radiative loss on gas-exposed metal (fluid AND solid).

    Surface flux q [W/m²] over one cell face removes q·dt/dx [J/m³] of
    enthalpy. (A previous version multiplied by ρ·cp as well, overweighting
    losses by ~six orders of magnitude.)
    """
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue

        n_exposed = 0
        if k + 1 < nz and flags[i, j, k + 1] == FLAG_GAS:
            n_exposed += 1
        if j + 1 < ny and flags[i, j + 1, k] == FLAG_GAS:
            n_exposed += 1
        if j - 1 >= 0 and flags[i, j - 1, k] == FLAG_GAS:
            n_exposed += 1
        if i + 1 < nx and flags[i + 1, j, k] == FLAG_GAS:
            n_exposed += 1
        if i - 1 >= 0 and flags[i - 1, j, k] == FLAG_GAS:
            n_exposed += 1

        if n_exposed == 0:
            continue

        T_c = T[i, j, k]
        T_eff = ti.min(T_c, 2500.0)
        q_loss = 0.0
        if enable_conv == 1:
            q_loss += h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_loss += eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)
        H[i, j, k] -= q_loss * ti.f32(n_exposed) * dt / dx


@ti.kernel
def clamp_enthalpy_floor(
    H: ti.template(),
    cp_rho_field: ti.template(),
    flags: ti.template(),
    T_floor: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h_min = cp_rho_field[i, j, k] * T_floor
        if H[i, j, k] < h_min:
            H[i, j, k] = h_min


@ti.kernel
def clamp_enthalpy_floor_scalar(
    H: ti.template(),
    cp_rho: ti.f32,
    flags: ti.template(),
    T_floor: ti.f32,
    FLAG_GAS: ti.i32,
):
    h_min = cp_rho * T_floor
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if H[i, j, k] < h_min:
            H[i, j, k] = h_min


@ti.kernel
def clamp_enthalpy_ceiling_scalar(
    H: ti.template(),
    flags: ti.template(),
    cp_rho: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    T_vapor_cap: ti.f32,
    L_rho: ti.f32,
    FLAG_GAS: ti.i32,
):
    """Cap enthalpy so recovered T ≤ T_vapor_cap (matches update_phase).

    Phase recovery uses H_liq = cp·T_solidus + L (not cp·T_liquidus + L).
    Ceiling must be H_liq + cp·(T_cap − T_liquidus), otherwise T overshoots
    the vapor cap by ~(T_liquidus − T_solidus) ≈ 45 K and the HUD sticks at
    ~2977 °C whenever the hotspot saturates.
    """
    H_sol = cp_rho * T_solidus
    H_liq = H_sol + L_rho
    h_cap_liquid = H_liq + cp_rho * (T_vapor_cap - T_liquidus)
    h_cap_solid = cp_rho * T_vapor_cap
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h = H[i, j, k]
        if h > H_liq:
            if h > h_cap_liquid:
                H[i, j, k] = h_cap_liquid
        else:
            if h > h_cap_solid:
                H[i, j, k] = h_cap_solid


@ti.kernel
def clamp_enthalpy_ceiling_variable_cp(
    H: ti.template(),
    flags: ti.template(),
    cp_rho_field: ti.template(),
    H_liq: ti.f32,
    T_liquidus: ti.f32,
    T_vapor_cap: ti.f32,
    FLAG_GAS: ti.i32,
):
    """Cap H using the same H_liq as update_phase_variable_cp."""
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        cp_r = cp_rho_field[i, j, k]
        h_cap_liquid = H_liq + cp_r * (T_vapor_cap - T_liquidus)
        h_cap_solid = cp_r * T_vapor_cap
        h = H[i, j, k]
        if h > H_liq:
            if h > h_cap_liquid:
                H[i, j, k] = h_cap_liquid
        else:
            if h > h_cap_solid:
                H[i, j, k] = h_cap_solid


@ti.kernel
def update_cooling_rate(
    T: ti.template(),
    T_prev: ti.template(),
    dT_dt: ti.template(),
    flags: ti.template(),
    dt_phys: ti.f32,
    FLAG_GAS: ti.i32,
):
    inv_dt = 1.0 / (dt_phys + 1e-12)
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        dT_dt[i, j, k] = (T[i, j, k] - T_prev[i, j, k]) * inv_dt
        T_prev[i, j, k] = T[i, j, k]


@ti.kernel
def apply_thermal_boundary_losses(
    H:    ti.template(),
    T:    ti.template(),
    flags: ti.template(),
    T_amb:     ti.f32,
    h_conv:    ti.f32,
    eps_rad:   ti.f32,
    enable_conv: ti.i32,
    enable_rad:  ti.i32,
    cp_rho:    ti.f32,
    dt:        ti.f32,
    dx:        ti.f32,
    sigma_sb:  ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Convective/radiative loss on gas-exposed metal (fluid AND solid).

    See apply_thermal_boundary_losses_variable for the flux-to-enthalpy
    conversion note (no ρ·cp factor — q·dt/dx is already J/m³).
    """
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue

        n_exposed = 0
        if k + 1 < nz and flags[i, j, k + 1] == FLAG_GAS:
            n_exposed += 1
        if j + 1 < ny and flags[i, j + 1, k] == FLAG_GAS:
            n_exposed += 1
        if j - 1 >= 0 and flags[i, j - 1, k] == FLAG_GAS:
            n_exposed += 1
        if i + 1 < nx and flags[i + 1, j, k] == FLAG_GAS:
            n_exposed += 1
        if i - 1 >= 0 and flags[i - 1, j, k] == FLAG_GAS:
            n_exposed += 1

        if n_exposed == 0:
            continue

        T_c = T[i, j, k]
        T_eff = ti.min(T_c, 2500.0)
        q_loss = 0.0
        if enable_conv == 1:
            q_loss += h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_loss += eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)

        H[i, j, k] -= q_loss * ti.f32(n_exposed) * dt / dx


@ti.kernel
def init_poiseuille_channel(
    f:     ti.template(),
    rho:   ti.template(),
    ux:    ti.template(),
    uy:    ti.template(),
    uz:    ti.template(),
    f_l:   ti.template(),
    phi:   ti.template(),
    flags: ti.template(),
    rho0:  ti.f32,
    ny:    ti.i32,
    nz:    ti.i32,
    fx_lu: ti.f32,
    nu_lu: ti.f32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
):
    """x-periodic channel with no-slip y/z walls and analytical Poiseuille equilibrium."""
    h_cells = ti.cast(ny - 2, ti.f32)

    for i, j, k in rho:
        if j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
            flags[i, j, k] = FLAG_SOLID
        else:
            flags[i, j, k] = FLAG_FLUID

        rho[i, j, k] = rho0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0
        f_l[i, j, k] = 1.0
        phi[i, j, k] = 1.0

        if flags[i, j, k] == FLAG_FLUID:
            y_from_wall = ti.cast(j - 1, ti.f32)
            ux_val = (fx_lu / (2.0 * nu_lu)) * y_from_wall * (h_cells - 1.0 - y_from_wall)
            ux[i, j, k] = ux_val
        else:
            ux[i, j, k] = 0.0

    for i, j, k in rho:
        if flags[i, j, k] != FLAG_FLUID:
            for q in ti.static(range(19)):
                f[q, i, j, k] = W[q] * rho0
        else:
            uxv = ux[i, j, k]
            uyv = uy[i, j, k]
            uzv = uz[i, j, k]
            u2 = uxv * uxv + uyv * uyv + uzv * uzv
            for q in ti.static(range(19)):
                eu = EX[q] * uxv + EY[q] * uyv + EZ[q] * uzv
                f[q, i, j, k] = W[q] * rho0 * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)


@ti.kernel
def set_uniform_Fx(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    fx_val: ti.f32,
    flags: ti.template(),
    FLAG_FLUID: ti.i32,
):
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_FLUID:
            Fx[i, j, k] = fx_val
            Fy[i, j, k] = 0.0
            Fz[i, j, k] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 5 — Marangoni CSF Force Assembly
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def compute_marangoni_force(
    T:    ti.template(),
    phi:  ti.template(),
    f_l:  ti.template(),
    Fx:   ti.template(),
    Fy:   ti.template(),
    Fz:   ti.template(),
    flags: ti.template(),
    dgamma_dT: ti.f32,      # dγ/dT already in lattice force units
    dx:        ti.f32,      # unused; kept for API stability
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Continuum Surface Force (CSF) Marangoni formulation (additive).

    τ_M = (dγ/dT) · ∇_s T
    F_M = (dγ/dT) · (I - n̂⊗n̂) · ∇T · |∇φ|

    Must use += so isotropic CSF / arc pressure / droplet loads survive.
    Never zeros F — only clear_forces may reset the force field.
    """
    eps = 1e-6
    fl_min = 0.05

    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < fl_min:
            continue

        dphi_x = 0.5 * (phi[ti.min(i+1,nx-1), j, k] - phi[ti.max(i-1,0), j, k])
        dphi_y = 0.5 * (phi[i, ti.min(j+1,ny-1), k] - phi[i, ti.max(j-1,0), k])
        dphi_z = 0.5 * (phi[i, j, ti.min(k+1,nz-1)] - phi[i, j, ti.max(k-1,0)])
        grad_phi_mag = ti.sqrt(dphi_x**2 + dphi_y**2 + dphi_z**2)

        if grad_phi_mag < eps:
            continue

        nx_n = dphi_x / grad_phi_mag
        ny_n = dphi_y / grad_phi_mag
        nz_n = dphi_z / grad_phi_mag

        dT_x = 0.5 * (T[ti.min(i+1,nx-1), j, k] - T[ti.max(i-1,0), j, k])
        dT_y = 0.5 * (T[i, ti.min(j+1,ny-1), k] - T[i, ti.max(j-1,0), k])
        dT_z = 0.5 * (T[i, j, ti.min(k+1,nz-1)] - T[i, j, ti.max(k-1,0)])

        n_dot_gradT = nx_n*dT_x + ny_n*dT_y + nz_n*dT_z
        dTs_x = dT_x - n_dot_gradT * nx_n
        dTs_y = dT_y - n_dot_gradT * ny_n
        dTs_z = dT_z - n_dot_gradT * nz_n

        scale = dgamma_dT * grad_phi_mag
        Fx[i, j, k] += scale * dTs_x
        Fy[i, j, k] += scale * dTs_y
        Fz[i, j, k] += scale * dTs_z


@ti.kernel
def compute_marangoni_force_variable(
    T: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    flags: ti.template(),
    dgamma_lu_field: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """CSF Marangoni with per-cell dγ/dT from material tables (additive)."""
    eps = 1e-6
    fl_min = 0.05
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < fl_min:
            continue

        dphi_x = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dphi_y = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dphi_z = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
        grad_phi_mag = ti.sqrt(dphi_x ** 2 + dphi_y ** 2 + dphi_z ** 2)

        if grad_phi_mag < eps:
            continue

        nx_n = dphi_x / grad_phi_mag
        ny_n = dphi_y / grad_phi_mag
        nz_n = dphi_z / grad_phi_mag

        dT_x = 0.5 * (T[ti.min(i + 1, nx - 1), j, k] - T[ti.max(i - 1, 0), j, k])
        dT_y = 0.5 * (T[i, ti.min(j + 1, ny - 1), k] - T[i, ti.max(j - 1, 0), k])
        dT_z = 0.5 * (T[i, j, ti.min(k + 1, nz - 1)] - T[i, j, ti.max(k - 1, 0)])

        n_dot_gradT = nx_n * dT_x + ny_n * dT_y + nz_n * dT_z
        dTs_x = dT_x - n_dot_gradT * nx_n
        dTs_y = dT_y - n_dot_gradT * ny_n
        dTs_z = dT_z - n_dot_gradT * nz_n

        scale = dgamma_lu_field[i, j, k] * grad_phi_mag
        Fx[i, j, k] += scale * dTs_x
        Fy[i, j, k] += scale * dTs_y
        Fz[i, j, k] += scale * dTs_z


# ──────────────────────────────────────────────────────────────────────────────
#  VOF — Phase field advection & flag update (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────
@ti.func
def _vof_face_flux(
    phi_src: ti.template(),
    vel: ti.template(),
    flags: ti.template(),
    i0: ti.i32, j0: ti.i32, k0: ti.i32,   # donor-side cell
    i1: ti.i32, j1: ti.i32, k1: ti.i32,   # receiver-side cell
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
) -> ti.f32:
    """Upwind volume flux through the face between (i0,j0,k0)→(i1,j1,k1).

    Face velocity is the average of the two adjacent cells, with SOLID and
    GAS cell velocities treated as zero (their stored u is stale/meaningless).
    No flux crosses a solid face. x wraps periodically (matching the LBM
    stream kernel) — clamping instead piled volume onto the outflow column
    where the [0,1] clamp deleted it (~5% loss over 80 steps at u=0.02).
    """
    flux = 0.0
    # Periodic in x, clamped (no-flux) in y/z — same BCs as `stream`.
    ii0 = (i0 + nx) % nx
    ii1 = (i1 + nx) % nx
    in0 = j0 >= 0 and k0 >= 0 and j0 < ny and k0 < nz
    in1 = j1 >= 0 and k1 >= 0 and j1 < ny and k1 < nz
    if in0 and in1:
        i0 = ii0
        i1 = ii1
        fl0 = flags[i0, j0, k0]
        fl1 = flags[i1, j1, k1]
        if fl0 != FLAG_SOLID and fl1 != FLAG_SOLID:
            u0 = 0.0
            u1 = 0.0
            if fl0 != FLAG_GAS:
                u0 = vel[i0, j0, k0]
            if fl1 != FLAG_GAS:
                u1 = vel[i1, j1, k1]
            u_face = 0.5 * (u0 + u1)
            if u_face > 0.0:
                flux = u_face * phi_src[i0, j0, k0]
            else:
                flux = u_face * phi_src[i1, j1, k1]
    return flux


@ti.kernel
def advect_phi(
    phi_dst: ti.template(),
    phi_src: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Conservative donor-cell (upwind flux-form) VOF advection.

    φ_new = φ − Σ_faces(out − in). Unlike the previous semi-Lagrangian
    round-to-nearest-cell pull (a no-op for |u| < 0.5 lu — i.e. always at
    melt-pool velocities), fractional volume moves every step and total metal
    volume is conserved to round-off away from domain boundaries. Gas cells
    may receive flux, becoming interface cells on the next flag update.
    """
    for i, j, k in phi_src:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID:
            phi_dst[i, j, k] = phi_src[i, j, k]
            continue

        f_e = _vof_face_flux(phi_src, ux, flags, i, j, k, i + 1, j, k,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)
        f_w = _vof_face_flux(phi_src, ux, flags, i - 1, j, k, i, j, k,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)
        f_n = _vof_face_flux(phi_src, uy, flags, i, j, k, i, j + 1, k,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)
        f_s = _vof_face_flux(phi_src, uy, flags, i, j - 1, k, i, j, k,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)
        f_t = _vof_face_flux(phi_src, uz, flags, i, j, k, i, j, k + 1,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)
        f_b = _vof_face_flux(phi_src, uz, flags, i, j, k - 1, i, j, k,
                             FLAG_SOLID, FLAG_GAS, nx, ny, nz)

        phi_dst[i, j, k] = phi_src[i, j, k] - (f_e - f_w) - (f_n - f_s) - (f_t - f_b)


@ti.kernel
def reinitialize_phi(
    phi: ti.template(),
    flags: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Clamp φ to [0,1] and enforce solid values.

    Gas cells KEEP incoming φ: the donor-cell flux uses zero velocity on the
    gas side, so φ cannot spread past the first gas layer, and any scrub
    threshold deletes exactly the volume advection just delivered (measured
    as a ~0.06%/step mass leak at melt-pool velocities). Only float noise
    (< 1e-6) is zeroed.
    """
    for i, j, k in phi:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID:
            phi[i, j, k] = 1.0
        elif flag == FLAG_GAS:
            if phi[i, j, k] < 1e-6:
                phi[i, j, k] = 0.0
            else:
                phi[i, j, k] = ti.min(1.0, phi[i, j, k])
        else:
            p = phi[i, j, k]
            phi[i, j, k] = ti.max(0.0, ti.min(1.0, p))


@ti.kernel
def update_flags_from_phi(
    phi: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    nz_solid: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
):
    """Derive FLUID / GAS / IFACE from φ and liquid fraction."""
    for i, j, k in phi:
        if k < nz_solid:
            flags[i, j, k] = FLAG_SOLID
            continue
        p = phi[i, j, k]
        fl = f_l[i, j, k]
        if p > 0.85 and fl < 0.05:
            flags[i, j, k] = FLAG_SOLID
        elif p > 0.55 and fl > 0.1:
            flags[i, j, k] = FLAG_FLUID
        elif p < 0.05:
            flags[i, j, k] = FLAG_GAS
        else:
            flags[i, j, k] = FLAG_IFACE


@ti.kernel
def solidify_cooled_metal(
    T: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    T_solidus: ti.f32,
    zero_velocity: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Promote solidified weld metal to static SOLID (bead growth / interpass)."""
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if phi[i, j, k] < 0.85:
            continue
        if f_l[i, j, k] < 0.02 and T[i, j, k] < T_solidus:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 0.0
            if zero_velocity == 1:
                ux[i, j, k] = 0.0
                uy[i, j, k] = 0.0
                uz[i, j, k] = 0.0


@ti.kernel
def remelt_hot_solid(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    cp_rho_field: ti.template(),
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    L_rho: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Re-melt deposited SOLID when interpass or new layer reheats above solidus."""
    margin = 5.0
    for i, j, k in T:
        if flags[i, j, k] != FLAG_SOLID:
            continue
        T_c = T[i, j, k]
        if T_c <= T_solidus + margin:
            continue
        cp_r = cp_rho_field[i, j, k]
        H_sol = cp_r * T_solidus
        fl_val = 0.0
        if T_c >= T_liquidus:
            fl_val = 1.0
            H[i, j, k] = cp_r * T_liquidus + L_rho + cp_r * (T_c - T_liquidus)
        else:
            fl_val = (T_c - T_solidus) / (T_liquidus - T_solidus + 1e-6)
            H[i, j, k] = H_sol + fl_val * L_rho
        f_l[i, j, k] = fl_val
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_FLUID


@ti.kernel
def remelt_hot_solid_scalar(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    cp_rho: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    L_rho: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
):
    margin = 5.0
    for i, j, k in T:
        if flags[i, j, k] != FLAG_SOLID:
            continue
        T_c = T[i, j, k]
        if T_c <= T_solidus + margin:
            continue
        H_sol = cp_rho * T_solidus
        fl_val = 0.0
        if T_c >= T_liquidus:
            fl_val = 1.0
            H[i, j, k] = cp_rho * T_liquidus + L_rho + cp_rho * (T_c - T_liquidus)
        else:
            fl_val = (T_c - T_solidus) / (T_liquidus - T_solidus + 1e-6)
            H[i, j, k] = H_sol + fl_val * L_rho
        f_l[i, j, k] = fl_val
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_FLUID


@ti.kernel
def solidify_trailing_pool(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    cp_rho_field: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    dir_x: ti.f32,
    dir_y: ti.f32,
    dir_z: ti.f32,
    lookback_cells: ti.f32,
    T_freeze: ti.f32,
    T_solidus: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Clamp the far trailing pool back to solid once it cools below a mild superheat margin."""
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if phi[i, j, k] < 0.85 or f_l[i, j, k] < 0.05:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        proj = di * dir_x + dj * dir_y + dk * dir_z
        if proj > -lookback_cells:
            continue
        if T[i, j, k] >= T_freeze:
            continue
        cp_r = cp_rho_field[i, j, k]
        # 1 K below solidus so HAZ time-above-solidus stops accumulating
        H[i, j, k] = cp_r * (T_solidus - 1.0)
        T[i, j, k] = T_solidus - 1.0
        f_l[i, j, k] = 0.0
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_SOLID
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0


@ti.kernel
def solidify_trailing_pool_scalar(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    cp_rho: ti.f32,
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    dir_x: ti.f32,
    dir_y: ti.f32,
    dir_z: ti.f32,
    lookback_cells: ti.f32,
    T_freeze: ti.f32,
    T_solidus: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if phi[i, j, k] < 0.85 or f_l[i, j, k] < 0.05:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        dk = ti.f32(k) - arc_k
        proj = di * dir_x + dj * dir_y + dk * dir_z
        if proj > -lookback_cells:
            continue
        if T[i, j, k] >= T_freeze:
            continue
        H[i, j, k] = cp_rho * (T_solidus - 1.0)
        T[i, j, k] = T_solidus - 1.0
        f_l[i, j, k] = 0.0
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_SOLID
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0


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
    T_amb: ti.f32,
    cp_rho: ti.f32,
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
            T[i, j, k] = T[si, j, k]
            H[i, j, k] = H[si, j, k]
            f_l[i, j, k] = f_l[si, j, k]
            phi[i, j, k] = phi[si, j, k]
            flags[i, j, k] = flags[si, j, k]
            T_max[i, j, k] = T_max[si, j, k]
            T_prev[i, j, k] = T_prev[si, j, k]
            dT_dt[i, j, k] = dT_dt[si, j, k]
            time_above_800[i, j, k] = time_above_800[si, j, k]
            time_above_1100[i, j, k] = time_above_1100[si, j, k]
            time_above_solidus[i, j, k] = time_above_solidus[si, j, k]
            rho[i, j, k] = rho[si, j, k]
            ux[i, j, k] = ux[si, j, k]
            uy[i, j, k] = uy[si, j, k]
            uz[i, j, k] = uz[si, j, k]
            Fx[i, j, k] = Fx[si, j, k]
            Fy[i, j, k] = Fy[si, j, k]
            Fz[i, j, k] = Fz[si, j, k]
            cp_rho_field[i, j, k] = cp_rho_field[si, j, k]
            alpha_lu_field[i, j, k] = alpha_lu_field[si, j, k]
            dgamma_lu_field[i, j, k] = dgamma_lu_field[si, j, k]
            tau_field[i, j, k] = tau_field[si, j, k]
            for q in ti.static(range(19)):
                f_a[q, i, j, k] = f_a[q, si, j, k]
                f_b[q, i, j, k] = f_b[q, si, j, k]

    i0 = nx - n_shift
    for i, j, k in ti.ndrange(nx, ny, nz):
        if i < i0:
            continue
        T[i, j, k] = T_amb
        H[i, j, k] = cp_rho * T_amb
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
        if k < nz_solid:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
        else:
            flags[i, j, k] = FLAG_GAS
            phi[i, j, k] = 0.0
        for q in ti.static(range(19)):
            feq = W[q] * rho0
            f_a[q, i, j, k] = feq
            f_b[q, i, j, k] = feq


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 6 — Boussinesq Buoyancy (add to Fz)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def add_buoyancy(
    T:    ti.template(),
    Fz:   ti.template(),
    f_l:  ti.template(),
    flags: ti.template(),
    g_lu:       ti.f32,   # Gravitational acceleration [lu/ts²]
    beta:       ti.f32,   # Thermal expansion coefficient [1/K]
    T_ref:      ti.f32,   # Reference temperature [K]
    rho_ref:    ti.f32,   # Reference density [lu]
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Boussinesq buoyancy: F_z = +ρ_lu·g·β·(T − T_ref) for liquid cells.
    Positive z is upward: hotter-than-reference liquid is lighter and must be
    pushed UP (the previous negative sign drove hot liquid down, inverting the
    convection cell). rho_ref is the lattice reference density (≈1).
    """
    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        fl = f_l[i, j, k]
        if fl > 0.0:
            Fz[i, j, k] += rho_ref * g_lu * beta * (T[i, j, k] - T_ref) * fl


@ti.kernel
def add_hydrostatic_gravity(
    Fz: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    rho_ref: ti.f32,
    g_lu: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Hydrostatic gravity on liquid: F_z = -ρ g f_l (+z up)."""
    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        fl = f_l[i, j, k]
        if fl > 0.0:
            Fz[i, j, k] += -rho_ref * g_lu * fl


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 7 — D3Q19 SRT Collision (V&V baseline)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def collide_srt(
    f_src: ti.template(),
    f_dst: ti.template(),
    rho:   ti.template(),
    ux:    ti.template(),
    uy:    ti.template(),
    uz:    ti.template(),
    Fx:    ti.template(),
    Fy:    ti.template(),
    Fz:    ti.template(),
    f_l:   ti.template(),
    flags: ti.template(),
    tau:         ti.f32,
    omega:       ti.f32,    # 1/tau
    dt_lu:       ti.f32,    # = 1.0 in LBM units
    C_darcy:     ti.f32,    # Carman-Kozeny constant
    FLAG_SOLID:  ti.i32,
    FLAG_GAS:    ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    D3Q19 SRT (BGK) collision with:
      - Guo forcing scheme (2002) for body forces
      - Semi-implicit Carman-Kozeny velocity correction for mushy zone

    Used as the V&V baseline. Production runs should use the MRT kernel.

    The Guo forcing scheme modifies the equilibrium to account for body forces:
      f_i^eq uses u* = u + F·τ/ρ  (velocity shift)
      After collision, u is corrected by + F·dt/(2ρ)

    Semi-implicit Carman-Kozeny:
      u_final = u* / (1 + dt·C·(1-fl)²/(fl³+ε))
    """
    eps = 1e-6

    for i, j, k in rho:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID or flag == FLAG_GAS:
            continue

        fl = f_l[i, j, k]

        # ── Moment extraction ─────────────────────────────────────────────
        r   = 0.0
        jx  = 0.0
        jy  = 0.0
        jz  = 0.0
        for q in ti.static(range(19)):
            fq  = f_src[q, i, j, k]
            r  += fq
            jx += fq * EX[q]
            jy += fq * EY[q]
            jz += fq * EZ[q]

        inv_r = 1.0 / (r + eps)

        # Guo forcing: shift velocity for equilibrium calculation
        fx = Fx[i, j, k]
        fy = Fy[i, j, k]
        fz = Fz[i, j, k]

        # Velocity before forcing correction
        ux_raw = jx * inv_r + 0.5 * fx * inv_r
        uy_raw = jy * inv_r + 0.5 * fy * inv_r
        uz_raw = jz * inv_r + 0.5 * fz * inv_r

        # Semi-implicit Carman-Kozeny drag correction
        ck_denom = 1.0 + C_darcy * ((1.0 - fl)**2) / (fl**3 + eps)
        ux_f = ux_raw / ck_denom
        uy_f = uy_raw / ck_denom
        uz_f = uz_raw / ck_denom

        # ── SRT Collision ─────────────────────────────────────────────────
        u2 = ux_f*ux_f + uy_f*uy_f + uz_f*uz_f
        for q in ti.static(range(19)):
            ex_q = EX[q]
            ey_q = EY[q]
            ez_q = EZ[q]
            eu   = ex_q*ux_f + ey_q*uy_f + ez_q*uz_f
            w_q  = W[q]

            # Equilibrium distribution
            feq = w_q * r * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2)

            # Guo forcing term. F is the lattice force density (ρ_lu ≈ 1 by
            # convention); the standard Guo source has NO 1/ρ factor — the
            # previous ·inv_r silently rescaled all body forces by 1/ρ_lu.
            F_dot_e = ex_q*fx + ey_q*fy + ez_q*fz
            S_q = w_q * (1.0 - 0.5*omega) * (
                3.0*(F_dot_e) +
                9.0*(eu * F_dot_e) -
                3.0*(ux_f*fx + uy_f*fy + uz_f*fz)
            )

            # Post-collision distribution (BGK + forcing)
            f_dst[q, i, j, k] = f_src[q, i, j, k] - omega * (
                f_src[q, i, j, k] - feq
            ) + S_q

        # ── Update macroscopic fields ──────────────────────────────────────
        rho[i, j, k] = r
        ux[i, j, k]  = ux_f
        uy[i, j, k]  = uy_f
        uz[i, j, k]  = uz_f


@ti.kernel
def collide_srt_variable_tau(
    f_src: ti.template(),
    f_dst: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    tau_field: ti.template(),
    C_darcy: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """SRT collision with per-cell relaxation time from μ(T) tables."""
    eps = 1e-6

    for i, j, k in rho:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID or flag == FLAG_GAS:
            continue

        fl = f_l[i, j, k]
        tau_loc = tau_field[i, j, k]
        omega = 1.0 / (tau_loc + eps)

        r = 0.0
        jx = 0.0
        jy = 0.0
        jz = 0.0
        for q in ti.static(range(19)):
            fq = f_src[q, i, j, k]
            r += fq
            jx += fq * EX[q]
            jy += fq * EY[q]
            jz += fq * EZ[q]

        inv_r = 1.0 / (r + eps)
        fx = Fx[i, j, k]
        fy = Fy[i, j, k]
        fz = Fz[i, j, k]
        ux_raw = jx * inv_r + 0.5 * fx * inv_r
        uy_raw = jy * inv_r + 0.5 * fy * inv_r
        uz_raw = jz * inv_r + 0.5 * fz * inv_r
        ck_denom = 1.0 + C_darcy * ((1.0 - fl) ** 2) / (fl ** 3 + eps)
        ux_f = ux_raw / ck_denom
        uy_f = uy_raw / ck_denom
        uz_f = uz_raw / ck_denom
        u2 = ux_f * ux_f + uy_f * uy_f + uz_f * uz_f

        for q in ti.static(range(19)):
            ex_q = EX[q]
            ey_q = EY[q]
            ez_q = EZ[q]
            eu = ex_q * ux_f + ey_q * uy_f + ez_q * uz_f
            w_q = W[q]
            feq = w_q * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
            F_dot_e = ex_q * fx + ey_q * fy + ez_q * fz
            S_q = w_q * (1.0 - 0.5 * omega) * (
                3.0 * F_dot_e
                + 9.0 * (eu * F_dot_e)
                - 3.0 * (ux_f * fx + uy_f * fy + uz_f * fz)
            )
            f_dst[q, i, j, k] = f_src[q, i, j, k] - omega * (
                f_src[q, i, j, k] - feq
            ) + S_q

        rho[i, j, k] = r
        ux[i, j, k] = ux_f
        uy[i, j, k] = uy_f
        uz[i, j, k] = uz_f


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 8 — Streaming (Pull Scheme + Bounce-Back)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def stream(
    f_src:  ti.template(),
    f_dst:  ti.template(),
    flags:  ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Pull-scheme streaming: each cell pulls its incoming populations from
    the upstream neighbour.

    Solid boundaries: bounce-back rule (no-slip condition).
    Gas cells: free-slip (zero-gradient) — Neumann BC.
    x-axis is periodic (required by the Poiseuille validation channel);
    y/z clamp (zero-gradient).
    """
    for i, j, k in flags:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID:
            continue

        for q in ti.static(range(19)):
            ni = i - EX[q]
            nj = j - EY[q]
            nk = k - EZ[q]

            ni_p = (ni + nx) % nx
            nj_p = ti.max(0, ti.min(nj, ny - 1))
            nk_p = ti.max(0, ti.min(nk, nz - 1))

            if flag == FLAG_GAS:
                f_dst[q, i, j, k] = f_src[q, ni_p, nj_p, nk_p]
            elif flags[ni_p, nj_p, nk_p] == FLAG_SOLID:
                f_dst[q, i, j, k] = f_src[OPP[q], i, j, k]
            else:
                f_dst[q, i, j, k] = f_src[q, ni_p, nj_p, nk_p]


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 9 — HAZ Tracker (Update T_max)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def update_T_max(
    T:     ti.template(),
    T_max: ti.template(),
    flags: ti.template(),
    FLAG_GAS: ti.i32,
):
    """Permanently record the peak temperature reached in each cell for HAZ tracking.
    Skips gas cells (T is meaningless there) for a minor GPU efficiency gain.
    Each (i,j,k) is unique per thread so the read-compare-write is race-free.
    """
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        current_T = T[i, j, k]
        if current_T > T_max[i, j, k]:
            T_max[i, j, k] = current_T


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 10 — Arc Pressure & Droplet Impact
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def apply_arc_pressure(
    Fz:    ti.template(),
    flags: ti.template(),
    phi:   ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma: ti.f32,
    pressure_pa: ti.f32,    # Peak arc pressure + droplet impact [Pa]
    dt: ti.f32,
    dx: ti.f32,
    rho_ref: ti.f32,        # Physical density [kg/m³]
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Apply a downward Gaussian Lorentz/Gas-shear force representing arc pressure
    and droplet impact to the free surface.
    
    The force is mapped from physical pressure [Pa] to LBM body force [lu/ts²].
    F_phys = P_phys / (rho * dx)  [m/s²]
    F_lu   = F_phys * dt² / dx    [lu/ts²]
    """
    eps = 1e-6
    # eps guards against a zero-sigma call producing inf in inv2s2
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)

    # Scale physical pressure to LBM force density
    F_peak_phys = pressure_pa / (rho_ref * dx)
    F_peak_lu   = F_peak_phys * dt**2 / dx

    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        # Apply pressure to free-surface interface cells.
        # Threshold 0.05 (down from 0.1) catches the full 2-cell-thick interface layer.
        dphi_z = phi[i, j, ti.min(k+1, Fz.shape[2]-1)] - phi[i, j, ti.max(k-1, 0)]
        if ti.abs(dphi_z) > 0.05:
            di = ti.f32(i) - arc_i
            dj = ti.f32(j) - arc_j
            dk = ti.f32(k) - arc_k
            r2 = di*di + dj*dj + dk*dk * 0.1
            force = -F_peak_lu * ti.math.exp(-r2 * inv2s2)
            Fz[i, j, k] += force


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL — Verification helpers
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def prescribe_gaussian_pulse(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    T_bg: ti.f32,
    A: ti.f32,
    sigma_m: ti.f32,
    cx: ti.f32,
    cy: ti.f32,
    cz: ti.f32,
    dx: ti.f32,
    cp_rho: ti.f32,
    FLAG_GAS: ti.i32,
):
    inv2s2 = 1.0 / (2.0 * sigma_m * sigma_m)
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        px = ti.f32(i) * dx - cx
        py = ti.f32(j) * dx - cy
        pz = ti.f32(k) * dx - cz
        r2 = px * px + py * py + pz * pz
        T_cell = T_bg + A * ti.math.exp(-r2 * inv2s2)
        T[i, j, k] = T_cell
        H[i, j, k] = cp_rho * T_cell
        f_l[i, j, k] = 1.0


@ti.kernel
def sync_T_from_H(H: ti.template(), T: ti.template(), cp_rho: ti.f32):
    for i, j, k in T:
        T[i, j, k] = H[i, j, k] / cp_rho


@ti.kernel
def init_stefan_liquid_column(
    T:     ti.template(),
    H:     ti.template(),
    f_l:   ti.template(),
    phi:   ti.template(),
    flags: ti.template(),
    nz_solid:   ti.i32,
    T_amb:      ti.f32,
    T_init:     ti.f32,
    cp_rho:     ti.f32,
    L_rho:      ti.f32,
    T_solidus:  ti.f32,
    T_liquidus: ti.f32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
):
    """1D Stefan setup: cold solid substrate below, superheated liquid above."""
    H_liq = cp_rho * T_liquidus + L_rho

    for i, j, k in T:
        if k < nz_solid:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 0.0
            T[i, j, k] = T_amb
            H[i, j, k] = cp_rho * T_amb
        else:
            flags[i, j, k] = FLAG_FLUID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 1.0
            T[i, j, k] = T_init
            H[i, j, k] = H_liq + cp_rho * (T_init - T_liquidus)


@ti.kernel
def clamp_substrate_enthalpy(
    H:     ti.template(),
    T:     ti.template(),
    f_l:   ti.template(),
    flags: ti.template(),
    nz_solid: ti.i32,
    T_amb:    ti.f32,
    cp_rho:   ti.f32,
    FLAG_SOLID: ti.i32,
):
    """Hold substrate at fixed temperature (Dirichlet BC for Stefan)."""
    for i, j, k in H:
        if k < nz_solid and flags[i, j, k] == FLAG_SOLID:
            H[i, j, k] = cp_rho * T_amb
            T[i, j, k] = T_amb
            f_l[i, j, k] = 0.0


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
    Advect active particles using the LBM fluid velocity.
    If a particle enters a solidifying cell (f_l < 0.05), it gets trapped.
    If it leaves the domain or enters gas, it becomes inactive.
    """
    nx, ny, nz = ux.shape
    for p in range(max_tracers):
        if active[p] == 1:
            # Current cell indices
            idx = pos[p] / dx
            i, j, k = ti.cast(idx.x, ti.i32), ti.cast(idx.y, ti.i32), ti.cast(idx.z, ti.i32)
            
            # Bounds check
            if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                active[p] = 0
                continue
                
            flag = flags[i, j, k]
            if flag == FLAG_GAS:
                active[p] = 0
                continue
                
            # If trapped in solid/mush, stop moving
            if f_l[i, j, k] < 0.05 or flag == FLAG_SOLID:
                # Active = 2 means permanently trapped (porosity)
                active[p] = 2
                continue
                
            # Nearest neighbor velocity (TODO: bilinear/trilinear interp)
            u = ti.Vector([ux[i, j, k], uy[i, j, k], uz[i, j, k]])
            
            # u is in lattice units (cells per timestep)
            # physical velocity = u * dx / dt
            # dx_phys = (u * dx / dt) * dt = u * dx
            pos[p] += u * dx


# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 12 — Diagnostics (HAZ time-at-T, export derived fields, force snapshot)
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def update_time_above_T(
    T: ti.template(),
    flags: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    dt: ti.f32,
    T_800: ti.f32,
    T_1100: ti.f32,
    T_solidus: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        t_c = T[i, j, k]
        if t_c >= T_800:
            time_above_800[i, j, k] += dt
        if t_c >= T_1100:
            time_above_1100[i, j, k] += dt
        if t_c >= T_solidus:
            time_above_solidus[i, j, k] += dt


@ti.kernel
def clamp_body_force_magnitude(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    flags: ti.template(),
    F_max_lu: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Limit |F| so Guo forcing cannot drive Ma ≫ 0.1 in one step."""
    eps = 1e-12
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        fx = Fx[i, j, k]
        fy = Fy[i, j, k]
        fz = Fz[i, j, k]
        mag = ti.sqrt(fx * fx + fy * fy + fz * fz)
        if mag > F_max_lu:
            s = F_max_lu / (mag + eps)
            Fx[i, j, k] = fx * s
            Fy[i, j, k] = fy * s
            Fz[i, j, k] = fz * s


@ti.kernel
def clamp_velocity_mach(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    u_max_lu: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Cap macroscopic lattice velocity (keeps VOF / telemetry physical)."""
    eps = 1e-12
    for i, j, k in ux:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        vx = ux[i, j, k]
        vy = uy[i, j, k]
        vz = uz[i, j, k]
        mag = ti.sqrt(vx * vx + vy * vy + vz * vz)
        if mag > u_max_lu:
            s = u_max_lu / (mag + eps)
            ux[i, j, k] = vx * s
            uy[i, j, k] = vy * s
            uz[i, j, k] = vz * s


@ti.kernel
def snapshot_forces(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    Fx_snap: ti.template(),
    Fy_snap: ti.template(),
    Fz_snap: ti.template(),
):
    for i, j, k in Fx:
        Fx_snap[i, j, k] = Fx[i, j, k]
        Fy_snap[i, j, k] = Fy[i, j, k]
        Fz_snap[i, j, k] = Fz[i, j, k]


@ti.kernel
def compute_curvature_field(
    phi: ti.template(),
    flags: ti.template(),
    kappa_out: ti.template(),
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """VOF curvature κ = -∇·n̂ (Brackbill; same as compute_csf_tension)."""
    eps = 1e-6
    for i, j, k in kappa_out:
        kappa_out[i, j, k] = 0.0
        if flags[i, j, k] == FLAG_GAS:
            continue
        if (
            i < 1 or j < 1 or k < 1
            or i > nx - 2 or j > ny - 2 or k > nz - 2
        ):
            continue
        kappa, gmag = _brackbill_curvature_at(phi, i, j, k, nx, ny, nz, eps)
        if gmag >= eps:
            kappa_out[i, j, k] = kappa


@ti.kernel
def compute_vorticity_magnitude(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    vort_out: ti.template(),
    dx: ti.f32,
    dt: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """|∇×u| in physical units [1/s] from lattice velocity.

    du_phys/dl_phys = (Δu_lu · dx/dt) / (Δcells · dx) = Δu_lu / (dt · Δcells)
    → the scale is 1/dt (the previous 1/(dx·dt) overstated vorticity by 1/dx).
    """
    scale = 1.0 / (dt + 1e-12)
    for i, j, k in vort_out:
        vort_out[i, j, k] = 0.0
        if flags[i, j, k] == FLAG_GAS:
            continue
        ip = ti.min(i + 1, nx - 1)
        im = ti.max(i - 1, 0)
        jp = ti.min(j + 1, ny - 1)
        jm = ti.max(j - 1, 0)
        kp = ti.min(k + 1, nz - 1)
        km = ti.max(k - 1, 0)
        duz_dy = 0.5 * (uz[i, jp, k] - uz[i, jm, k])
        duy_dz = 0.5 * (uy[i, j, kp] - uy[i, j, km])
        dux_dz = 0.5 * (ux[i, j, kp] - ux[i, j, km])
        duz_dx = 0.5 * (uz[ip, j, k] - uz[im, j, k])
        duy_dx = 0.5 * (uy[ip, j, k] - uy[im, j, k])
        dux_dy = 0.5 * (ux[i, jp, k] - ux[i, jm, k])
        wx = (duz_dy - duy_dz) * scale
        wy = (dux_dz - duz_dx) * scale
        wz = (duy_dx - dux_dy) * scale
        vort_out[i, j, k] = ti.sqrt(wx * wx + wy * wy + wz * wz)


# ──────────────────────────────────────────────────────────────────────────────
#  Telemetry reductions + φ ↔ f_l consistency
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def telemetry_pool_reduce(
    T: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    dT_dt: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    n_buf: ti.template(),
    T_peak_buf: ti.template(),
    cool_buf: ti.template(),
    u2_buf: ti.template(),
    i_sum_buf: ti.template(),
    i_min_buf: ti.template(),
    i_max_buf: ti.template(),
    T_global_buf: ti.template(),
):
    """GPU-side pool aggregates — avoids copying entire volumes for telemetry.

    Peak cooling rate uses all metal (φ > 0.05), not only liquid — otherwise
    it stays 0 after the bead freezes even while the HAZ is still cooling.
    """
    n_buf[None] = 0
    T_peak_buf[None] = 0.0
    cool_buf[None] = 0.0
    u2_buf[None] = 0.0
    i_sum_buf[None] = 0.0
    i_min_buf[None] = 1.0e9
    i_max_buf[None] = -1.0e9
    T_global_buf[None] = 0.0
    for i, j, k in T:
        Tc = T[i, j, k]
        ti.atomic_max(T_global_buf[None], Tc)
        if phi[i, j, k] > 0.05:
            # Positive while cooling (metallurgical sign).
            ti.atomic_max(cool_buf[None], -dT_dt[i, j, k])
        if f_l[i, j, k] > 0.5:
            ti.atomic_add(n_buf[None], 1)
            ti.atomic_max(T_peak_buf[None], Tc)
            u2 = ux[i, j, k] * ux[i, j, k] + uy[i, j, k] * uy[i, j, k] + uz[i, j, k] * uz[i, j, k]
            ti.atomic_max(u2_buf[None], u2)
            ti.atomic_add(i_sum_buf[None], ti.f32(i))
            ti.atomic_min(i_min_buf[None], ti.f32(i))
            ti.atomic_max(i_max_buf[None], ti.f32(i))


@ti.kernel
def extract_fl_yz_slice(
    f_l: ti.template(),
    out: ti.template(),
    i_slice: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Copy one y–z face of f_l for host-side pool W/D measurement."""
    for j, k in ti.ndrange(ny, nz):
        out[j, k] = f_l[i_slice, j, k]


@ti.kernel
def sync_phi_liquid_fraction(
    phi: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
):
    """Keep φ (metal presence) and f_l (melt state) from drifting apart.

    - Pure gas (φ < 0.05 or FLAG_GAS): f_l must be 0 — no liquid fraction in air.
    - Solid metal (FLAG_SOLID): f_l forced to 0 (already solidified).
    - Metal with φ ≥ 0.55 and f_l < 0 but T-driven melt may set f_l later;
      here we only clear contradictory liquid-in-gas.
    """
    for i, j, k in phi:
        fl = flags[i, j, k]
        if fl == FLAG_GAS or phi[i, j, k] < 0.05:
            f_l[i, j, k] = 0.0
        elif fl == FLAG_SOLID:
            f_l[i, j, k] = 0.0
