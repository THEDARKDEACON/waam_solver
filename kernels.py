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
#  D3Q19 velocity-set constants (module-level ti.field refs set by grid.py)
# ──────────────────────────────────────────────────────────────────────────────
# These are set by WAAMGrid after field allocation so that kernels share the
# same constant arrays without copying.  They are module-level so any kernel
# can reference them without closures.

EX  = None  # ti.field(int,  shape=(19,))
EY  = None
EZ  = None
W   = None  # ti.field(f32, shape=(19,))
OPP = None  # ti.field(int,  shape=(19,))

def bind_velocity_set(grid):
    """Bind module-level velocity set fields from the allocated grid."""
    global EX, EY, EZ, W, OPP
    EX  = grid.ex
    EY  = grid.ey
    EZ  = grid.ez
    W   = grid.w
    OPP = grid.opp


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
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Fill the entire grid with a quiescent, ambient-temperature initial state.
    Bottom `nz_solid` layers are set as solid substrate.
    Top layers above the liquid are gas cells.
    """
    for q, i, j, k in f:
        w_q = W[q]
        f[q, i, j, k] = w_q * rho0   # Equilibrium at rest → f_eq = w * rho

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
            phi[i, j, k] = 1.0 if k < nz_solid else 0.0
            if k < nz_solid:
                flags[i, j, k] = FLAG_SOLID
            else:
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
            flags[i, j, k] = FLAG_FLUID
            f_l[i, j, k] = 1.0
            phi[i, j, k] = 1.0
            T[i, j, k] = T_drop
            H[i, j, k] = cp_rho * T_drop + L_rho
            rho[i, j, k] = rho0
            ti.atomic_add(vol_acc[None], cell_vol)
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
    Distribute arc heat via a 2D Gaussian flux profile on the free surface.

    Energy is injected into the enthalpy field H (not T directly), ensuring
    that phase-change latent heat is correctly absorbed before T rises.

    Gaussian: q(r) = Q·η / (2π·σ²) · exp(-r² / (2σ²))
    Total energy added per timestep to a cell: ΔH = q(r) · dt / dx³
    """
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    norm   = Q_w * eta / (2.0 * ti.math.pi * sigma * sigma)

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
        flux = norm * ti.math.exp(-r2 * inv2s2) * w_dep
        H[i, j, k] += flux * dt / dx3


@ti.kernel
def inject_goldak_heat(
    H: ti.template(),
    flags: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    Q_w: ti.f32,
    sigma: ti.f32,
    dt: ti.f32,
    dx3: ti.f32,
    eta: ti.f32,
    travel_sign: ti.f32,
    ff: ti.f32,
    fr: ti.f32,
    depth_front: ti.f32,
    depth_rear: ti.f32,
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """
    Goldak double-ellipsoid heat source (simplified 3D).

    Leading (front) and trailing (rear) ellipsoids share transverse semi-axes
    σ but differ in depth.  travel_sign = +1 when torch moves +x.
    """
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)

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
        r2_xy = di * di + dj * dj

        is_front = (di * travel_sign) >= 0.0
        depth = ti.select(is_front, depth_front, depth_rear)
        frac = ti.select(is_front, ff, fr)
        dk_eff = dk / (depth + eps)
        r2 = r2_xy + dk_eff * dk_eff * 4.0

        norm = Q_w * eta * frac / (2.0 * ti.math.pi * sigma * sigma + eps)
        flux = norm * ti.math.exp(-r2 * inv2s2) * w_dep
        H[i, j, k] += flux * dt / dx3


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
):
    """
    Balanced-force CSF surface tension: F = γ κ ∇φ with κ = -∇·n̂.
    """
    eps = 1e-6
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        dpx = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dpy = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dpz = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
        gmag = ti.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
        if gmag < eps:
            continue

        nx_n = dpx / gmag
        ny_n = dpy / gmag
        nz_n = dpz / gmag

        div_n = (
            0.5 * (
                (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
                - (phi[ti.max(i - 1, 0), j, k] - phi[ti.max(i - 2, 0), j, k])
            )
            + 0.5 * (
                (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
                - (phi[i, ti.max(j - 1, 0), k] - phi[i, ti.max(j - 2, 0), k])
            )
            + 0.5 * (
                (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
                - (phi[i, j, ti.max(k - 1, 0)] - phi[i, j, ti.max(k - 2, 0)])
            )
        )
        kappa = -div_n / (gmag + eps)
        scale = gamma_lu * kappa * gmag
        Fx[i, j, k] += scale * dpx
        Fy[i, j, k] += scale * dpy
        Fz[i, j, k] += scale * dpz


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
#  KERNEL 4 — Thermal Advection-Diffusion (Upwind Scheme)
# ──────────────────────────────────────────────────────────────────────────────
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
    Solve ∂H/∂t + u·∇H = α·∇²H in lattice units.

    Uses a first-order upwind scheme for advection (TVD is TODO).
    Diffusion uses central differences.
    """
    for i, j, k in H:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        T_c  = T[i, j, k]
        u    = ux[i, j, k]
        v    = uy[i, j, k]
        w    = uz[i, j, k]

        # --- Upwind advection (ti.select avoids kernel name errors) ---
        T_im = T[ti.max(i-1, 0), j, k]
        T_ip = T[ti.min(i+1, nx-1), j, k]
        T_jm = T[i, ti.max(j-1, 0), k]
        T_jp = T[i, ti.min(j+1, ny-1), k]
        T_km = T[i, j, ti.max(k-1, 0)]
        T_kp = T[i, j, ti.min(k+1, nz-1)]

        dTdx = ti.select(u > 0.0, T_c - T_im, T_ip - T_c)
        dTdy = ti.select(v > 0.0, T_c - T_jm, T_jp - T_c)
        dTdz = ti.select(w > 0.0, T_c - T_km, T_kp - T_c)

        advection = u * dTdx + v * dTdy + w * dTdz

        # --- Central-difference diffusion ---
        lap = (
            T_ip + T_im +
            T_jp + T_jm +
            T_kp + T_km -
            6.0 * T_c
        )

        dT = (-advection + alpha_lu * lap) * dt
        H[i, j, k] += cp_rho * dT


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
            dgamma_lu_field[i, j, k] = dgamma * force_scale
            tau_field[i, j, k] = 3.0 * nu_lu + 0.5
        else:
            cp_rho_field[i, j, k] = cp_rho_ref
            alpha_lu_field[i, j, k] = alpha_lu_ref
            dgamma_lu_field[i, j, k] = dgamma_lu_ref
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
    for i, j, k in H:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        T_c = T[i, j, k]
        alpha_lu = alpha_lu_field[i, j, k]
        cp_rho = cp_rho_field[i, j, k]
        u = ux[i, j, k]
        v = uy[i, j, k]
        w = uz[i, j, k]

        T_im = T[ti.max(i - 1, 0), j, k]
        T_ip = T[ti.min(i + 1, nx - 1), j, k]
        T_jm = T[i, ti.max(j - 1, 0), k]
        T_jp = T[i, ti.min(j + 1, ny - 1), k]
        T_km = T[i, j, ti.max(k - 1, 0)]
        T_kp = T[i, j, ti.min(k + 1, nz - 1)]

        dTdx = ti.select(u > 0.0, T_c - T_im, T_ip - T_c)
        dTdy = ti.select(v > 0.0, T_c - T_jm, T_jp - T_c)
        dTdz = ti.select(w > 0.0, T_c - T_km, T_kp - T_c)
        advection = u * dTdx + v * dTdy + w * dTdz

        lap = T_ip + T_im + T_jp + T_jm + T_kp + T_km - 6.0 * T_c
        dT = (-advection + alpha_lu * lap) * dt
        H[i, j, k] += cp_rho * dT


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
    for i, j, k in H:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        exposed = False
        if k + 1 < nz and flags[i, j, k + 1] == FLAG_GAS:
            exposed = True
        if j + 1 < ny and flags[i, j + 1, k] == FLAG_GAS:
            exposed = True
        if j - 1 >= 0 and flags[i, j - 1, k] == FLAG_GAS:
            exposed = True
        if i + 1 < nx and flags[i + 1, j, k] == FLAG_GAS:
            exposed = True
        if i - 1 >= 0 and flags[i - 1, j, k] == FLAG_GAS:
            exposed = True

        if not exposed:
            continue

        T_c = T[i, j, k]
        T_eff = ti.min(T_c, 2500.0)
        cp_rho = cp_rho_field[i, j, k]
        q_loss = 0.0
        if enable_conv == 1:
            q_loss += h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_loss += eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)
        H[i, j, k] -= q_loss * dt / dx * cp_rho


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
    """Cap enthalpy at vaporization ceiling (prevents runaway superheat)."""
    h_liq = cp_rho * T_liquidus + L_rho
    h_cap_liquid = h_liq + cp_rho * (T_vapor_cap - T_liquidus)
    h_cap_solid = cp_rho * T_vapor_cap
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h = H[i, j, k]
        if h > h_liq:
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
    T_liquidus: ti.f32,
    T_vapor_cap: ti.f32,
    L_rho: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        cp_r = cp_rho_field[i, j, k]
        h_liq = cp_r * T_liquidus + L_rho
        h_cap_liquid = h_liq + cp_r * (T_vapor_cap - T_liquidus)
        h_cap_solid = cp_r * T_vapor_cap
        h = H[i, j, k]
        if h > h_liq:
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
    """Convective/radiative heat loss on cells adjacent to gas."""
    for i, j, k in H:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        exposed = False
        if k + 1 < nz and flags[i, j, k + 1] == FLAG_GAS:
            exposed = True
        if j + 1 < ny and flags[i, j + 1, k] == FLAG_GAS:
            exposed = True
        if j - 1 >= 0 and flags[i, j - 1, k] == FLAG_GAS:
            exposed = True
        if i + 1 < nx and flags[i + 1, j, k] == FLAG_GAS:
            exposed = True
        if i - 1 >= 0 and flags[i - 1, j, k] == FLAG_GAS:
            exposed = True

        if not exposed:
            continue

        T_c = T[i, j, k]
        T_eff = ti.min(T_c, 2500.0)
        q_loss = 0.0
        if enable_conv == 1:
            q_loss += h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_loss += eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)

        H[i, j, k] -= q_loss * dt / dx * cp_rho


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

    for q, i, j, k in f:
        if flags[i, j, k] != FLAG_FLUID:
            f[q, i, j, k] = W[q] * rho0
            continue
        uxv = ux[i, j, k]
        uyv = uy[i, j, k]
        uzv = uz[i, j, k]
        u2 = uxv * uxv + uyv * uyv + uzv * uzv
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
    dgamma_dT: ti.f32,      # dγ/dT  [N/(m·K)] in LBM units
    dx:        ti.f32,      # Cell size [m] (for force scaling)
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Continuum Surface Force (CSF) Marangoni formulation.

    The Marangoni shear stress at the free surface is:
      τ_M = (dγ/dT) · ∇_s T   [N/m²]

    Translated to a CSF body force:
      F_M = (dγ/dT) · (I - n̂⊗n̂) · ∇T · |∇φ|

    where n̂ = ∇φ / |∇φ| is the interface normal from the VOF field.

    For simplicity at this stage, we apply a surface-tangential
    temperature gradient force to cells near the interface (|∇φ| > ε).
    """
    eps = 1e-6

    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            Fx[i, j, k] = 0.0
            Fy[i, j, k] = 0.0
            Fz[i, j, k] = 0.0
            continue

        # VOF gradient (interface normal direction)
        dphi_x = 0.5 * (phi[ti.min(i+1,nx-1), j, k] - phi[ti.max(i-1,0), j, k])
        dphi_y = 0.5 * (phi[i, ti.min(j+1,ny-1), k] - phi[i, ti.max(j-1,0), k])
        dphi_z = 0.5 * (phi[i, j, ti.min(k+1,nz-1)] - phi[i, j, ti.max(k-1,0)])
        grad_phi_mag = ti.sqrt(dphi_x**2 + dphi_y**2 + dphi_z**2)

        if grad_phi_mag < eps:
            Fx[i, j, k] = 0.0
            Fy[i, j, k] = 0.0
            Fz[i, j, k] = 0.0
            continue

        # Interface normal
        nx_n = dphi_x / grad_phi_mag
        ny_n = dphi_y / grad_phi_mag
        nz_n = dphi_z / grad_phi_mag

        # Temperature gradient
        dT_x = 0.5 * (T[ti.min(i+1,nx-1), j, k] - T[ti.max(i-1,0), j, k])
        dT_y = 0.5 * (T[i, ti.min(j+1,ny-1), k] - T[i, ti.max(j-1,0), k])
        dT_z = 0.5 * (T[i, j, ti.min(k+1,nz-1)] - T[i, j, ti.max(k-1,0)])

        # Project ∇T onto the interface tangent plane: ∇_s T = (I - n̂⊗n̂)·∇T
        n_dot_gradT = nx_n*dT_x + ny_n*dT_y + nz_n*dT_z
        dTs_x = dT_x - n_dot_gradT * nx_n
        dTs_y = dT_y - n_dot_gradT * ny_n
        dTs_z = dT_z - n_dot_gradT * nz_n

        # CSF body force: F = (dγ/dT) · ∇_s T · |∇φ|
        scale = dgamma_dT * grad_phi_mag
        Fx[i, j, k] = scale * dTs_x
        Fy[i, j, k] = scale * dTs_y
        Fz[i, j, k] = scale * dTs_z


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
    """CSF Marangoni with per-cell dγ/dT from material tables."""
    eps = 1e-6
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            Fx[i, j, k] = 0.0
            Fy[i, j, k] = 0.0
            Fz[i, j, k] = 0.0
            continue

        dphi_x = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dphi_y = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dphi_z = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
        grad_phi_mag = ti.sqrt(dphi_x ** 2 + dphi_y ** 2 + dphi_z ** 2)

        if grad_phi_mag < eps:
            Fx[i, j, k] = 0.0
            Fy[i, j, k] = 0.0
            Fz[i, j, k] = 0.0
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
        Fx[i, j, k] = scale * dTs_x
        Fy[i, j, k] = scale * dTs_y
        Fz[i, j, k] = scale * dTs_z


# ──────────────────────────────────────────────────────────────────────────────
#  VOF — Phase field advection & flag update (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────
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
    """Donor-cell (first-order) VOF advection in lattice units."""
    for i, j, k in phi_src:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID:
            phi_dst[i, j, k] = phi_src[i, j, k]
            continue
        if flag == FLAG_GAS:
            phi_dst[i, j, k] = 0.0
            continue

        u = ux[i, j, k]
        v = uy[i, j, k]
        w = uz[i, j, k]
        ib = ti.cast(ti.round(ti.f32(i) - u), ti.i32)
        jb = ti.cast(ti.round(ti.f32(j) - v), ti.i32)
        kb = ti.cast(ti.round(ti.f32(k) - w), ti.i32)
        ib = ti.max(0, ti.min(ib, nx - 1))
        jb = ti.max(0, ti.min(jb, ny - 1))
        kb = ti.max(0, ti.min(kb, nz - 1))
        phi_dst[i, j, k] = phi_src[ib, jb, kb]


@ti.kernel
def reinitialize_phi(
    phi: ti.template(),
    flags: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Clamp φ to [0,1] and enforce gas/solid values."""
    for i, j, k in phi:
        flag = flags[i, j, k]
        if flag == FLAG_SOLID:
            phi[i, j, k] = 1.0
        elif flag == FLAG_GAS:
            phi[i, j, k] = 0.0
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
        if p > 0.55 and fl > 0.1:
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
    T_solidus: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Promote solidified weld metal to static SOLID (bead growth / interpass)."""
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.02 and T[i, j, k] < T_solidus:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 0.0


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
    """Shift all fields left by n_shift cells; reset right strip to ambient substrate/gas."""
    for j, k in ti.ndrange(ny, nz):
        for ii in range(nx - n_shift):
            i = nx - 1 - n_shift - ii
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
    Boussinesq buoyancy: F_z = -ρ·g·β·(T - T_ref) for liquid cells only.
    Positive z is upward; hot liquid rises.
    """
    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        fl = f_l[i, j, k]
        if fl > 0.0:
            Fz[i, j, k] += -rho_ref * g_lu * beta * (T[i, j, k] - T_ref) * fl


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

            # Guo forcing term
            F_dot_e = ex_q*fx + ey_q*fy + ez_q*fz
            S_q = w_q * (1.0 - 0.5*omega) * (
                3.0*(F_dot_e) +
                9.0*(eu * F_dot_e) -
                3.0*(ux_f*fx + uy_f*fy + uz_f*fz)
            ) * inv_r

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
            ) * inv_r
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
    """
    for q, i, j, k in f_src:
        flag = flags[i, j, k]

        ni = i - EX[q]
        nj = j - EY[q]
        nk = k - EZ[q]

        ni_p = (ni + nx) % nx
        nj_p = ti.max(0, ti.min(nj, ny - 1))
        nk_p = ti.max(0, ti.min(nk, nz - 1))

        src_flag = flags[ni_p, nj_p, nk_p]

        if flag == FLAG_SOLID:
            continue

        if flag == FLAG_GAS:
            f_dst[q, i, j, k] = f_src[q, ni_p, nj_p, nk_p]
            continue

        if src_flag == FLAG_SOLID:
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
    """VOF curvature κ = -∇·n̂ (same convention as compute_csf_tension)."""
    eps = 1e-6
    for i, j, k in kappa_out:
        kappa_out[i, j, k] = 0.0
        if flags[i, j, k] == FLAG_GAS:
            continue
        dpx = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dpy = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dpz = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
        gmag = ti.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
        if gmag < eps:
            continue
        div_n = (
            0.5 * (
                (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
                - (phi[ti.max(i - 1, 0), j, k] - phi[ti.max(i - 2, 0), j, k])
            )
            + 0.5 * (
                (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
                - (phi[i, ti.max(j - 1, 0), k] - phi[i, ti.max(j - 2, 0), k])
            )
            + 0.5 * (
                (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
                - (phi[i, j, ti.max(k - 1, 0)] - phi[i, j, ti.max(k - 2, 0)])
            )
        )
        kappa_out[i, j, k] = -div_n / (gmag + eps)


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
    """|∇×u| in physical units [1/s] from lattice velocity."""
    scale = 1.0 / (dx + 1e-12) / (dt + 1e-12)
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
