"""Taichi kernels — heat injection, enthalpy, losses, phase."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    _alloy_pick,
    lookup_table_1d,
)
from .grid_init import arc_deposition_weight

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
    dir_x: ti.f32,
    dir_y: ti.f32,
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
    Travel frame: (dir_x, dir_y) is the unit travel direction in the XY plane
    (from torch path). x_travel ≥ 0 ⇒ front ellipsoid.
    """
    eps = 1e-6
    x_travel = di * dir_x + dj * dir_y
    y_trans = -di * dir_y + dj * dir_x
    is_front = x_travel >= 0.0
    a_axis = ti.select(is_front, a_front, a_rear)
    frac = ti.select(is_front, ff, fr)
    return frac * ti.math.exp(
        -3.0 * (
            (x_travel * x_travel) / (a_axis * a_axis + eps)
            + (y_trans * y_trans) / (b_axis * b_axis + eps)
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
    dir_x: ti.f32,
    dir_y: ti.f32,
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
    (dir_x, dir_y) is the horizontal unit travel direction.
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
            di, dj, dk, dir_x, dir_y, ff, fr,
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
            di, dj, dk, dir_x, dir_y, ff, fr,
            a_front, a_rear, b_axis, c_axis,
        )
        H[i, j, k] += energy_per_weight * w * w_dep

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
def refresh_thermal_properties_dual(
    T: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    dgamma_lu_field: ti.template(),
    tau_field: ti.template(),
    alloy_id: ti.template(),
    flags: ti.template(),
    w_cp_T: ti.template(),
    w_cp_V: ti.template(),
    w_k_T: ti.template(),
    w_k_V: ti.template(),
    w_mu_T: ti.template(),
    w_mu_V: ti.template(),
    w_dgamma_T: ti.template(),
    w_dgamma_V: ti.template(),
    w_n_cp: ti.template(),
    w_n_k: ti.template(),
    w_n_mu: ti.template(),
    w_n_dgamma: ti.template(),
    p_cp_T: ti.template(),
    p_cp_V: ti.template(),
    p_k_T: ti.template(),
    p_k_V: ti.template(),
    p_mu_T: ti.template(),
    p_mu_V: ti.template(),
    p_dgamma_T: ti.template(),
    p_dgamma_V: ti.template(),
    p_n_cp: ti.template(),
    p_n_k: ti.template(),
    p_n_mu: ti.template(),
    p_n_dgamma: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
    rho_w: ti.f32,
    rho_p: ti.f32,
    w_cp_fb: ti.f32,
    w_k_fb: ti.f32,
    w_mu_fb: ti.f32,
    w_dgamma_fb: ti.f32,
    p_cp_fb: ti.f32,
    p_k_fb: ti.f32,
    p_mu_fb: ti.f32,
    p_dgamma_fb: ti.f32,
    force_scale_w: ti.f32,
    force_scale_p: ti.f32,
    cp_rho_w: ti.f32,
    cp_rho_p: ti.f32,
    alpha_lu_w: ti.f32,
    alpha_lu_p: ti.f32,
    dgamma_lu_w: ti.f32,
    dgamma_lu_p: ti.f32,
    tau_w: ti.f32,
    tau_p: ti.f32,
    marangoni_scale: ti.f32,
    use_tables_w: ti.i32,
    use_tables_p: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Per-cell wire/plate tables. Solid plate conducts with plate k(T), cp(T)."""
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            cp_rho_field[i, j, k] = cp_rho_w
            alpha_lu_field[i, j, k] = alpha_lu_w
            dgamma_lu_field[i, j, k] = dgamma_lu_w
            tau_field[i, j, k] = tau_w
            continue
        frac = alloy_id[i, j, k]
        T_c = T[i, j, k]
        w_cp_r = cp_rho_w
        w_alpha = alpha_lu_w
        w_dgamma = dgamma_lu_w * marangoni_scale
        w_tau = tau_w
        p_cp_r = cp_rho_p
        p_alpha = alpha_lu_p
        p_dgamma = dgamma_lu_p * marangoni_scale
        p_tau = tau_p
        if use_tables_w == 1:
            w_cp = lookup_table_1d(T_c, w_cp_T, w_cp_V, w_n_cp[None], w_cp_fb)
            w_k = lookup_table_1d(T_c, w_k_T, w_k_V, w_n_k[None], w_k_fb)
            w_mu = lookup_table_1d(T_c, w_mu_T, w_mu_V, w_n_mu[None], w_mu_fb)
            w_dg = lookup_table_1d(T_c, w_dgamma_T, w_dgamma_V, w_n_dgamma[None], w_dgamma_fb)
            w_cp_r = rho_w * w_cp
            w_alpha = (w_k / (rho_w * w_cp + 1e-9)) * dt / (dx * dx)
            w_dgamma = w_dg * force_scale_w * marangoni_scale
            w_tau = 3.0 * ((w_mu / (rho_w + 1e-9)) * dt / (dx * dx)) + 0.5
        if use_tables_p == 1:
            p_cp = lookup_table_1d(T_c, p_cp_T, p_cp_V, p_n_cp[None], p_cp_fb)
            p_k = lookup_table_1d(T_c, p_k_T, p_k_V, p_n_k[None], p_k_fb)
            p_mu = lookup_table_1d(T_c, p_mu_T, p_mu_V, p_n_mu[None], p_mu_fb)
            p_dg = lookup_table_1d(T_c, p_dgamma_T, p_dgamma_V, p_n_dgamma[None], p_dgamma_fb)
            p_cp_r = rho_p * p_cp
            p_alpha = (p_k / (rho_p * p_cp + 1e-9)) * dt / (dx * dx)
            p_dgamma = p_dg * force_scale_p * marangoni_scale
            p_tau = 3.0 * ((p_mu / (rho_p + 1e-9)) * dt / (dx * dx)) + 0.5
        om = 1.0 - frac
        cp_rho_field[i, j, k] = om * w_cp_r + frac * p_cp_r
        alpha_lu_field[i, j, k] = om * w_alpha + frac * p_alpha
        dgamma_lu_field[i, j, k] = om * w_dgamma + frac * p_dgamma
        tau_field[i, j, k] = om * w_tau + frac * p_tau


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
def update_phase_dual(
    H: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    cp_rho_field: ti.template(),
    alloy_id: ti.template(),
    L_rho_w: ti.f32,
    T_sol_w: ti.f32,
    T_liq_w: ti.f32,
    H_sol_w: ti.f32,
    H_liq_w: ti.f32,
    L_rho_p: ti.f32,
    T_sol_p: ti.f32,
    T_liq_p: ti.f32,
    H_sol_p: ti.f32,
    H_liq_p: ti.f32,
):
    """Enthalpy–porosity recovery with composition-weighted melting range."""
    for i, j, k in H:
        h = H[i, j, k]
        cp_rho = cp_rho_field[i, j, k]
        L_rho = _alloy_pick(alloy_id[i, j, k], L_rho_w, L_rho_p)
        T_sol = _alloy_pick(alloy_id[i, j, k], T_sol_w, T_sol_p)
        T_liq = _alloy_pick(alloy_id[i, j, k], T_liq_w, T_liq_p)
        H_sol = _alloy_pick(alloy_id[i, j, k], H_sol_w, H_sol_p)
        H_liq = _alloy_pick(alloy_id[i, j, k], H_liq_w, H_liq_p)
        if h <= H_sol:
            f_l[i, j, k] = 0.0
            T[i, j, k] = h / (cp_rho + 1e-9)
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
            T[i, j, k] = T_liq + (h - H_liq) / (cp_rho + 1e-9)
        else:
            fl = (h - H_sol) / L_rho
            f_l[i, j, k] = fl
            T[i, j, k] = T_sol + fl * (T_liq - T_sol)


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
    T_cap_K: ti.f32,
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

    T_cap_K is an f32 safety ceiling (typically T_vapor_cap), not a 2500 K
    physics floor — pool peaks near boiling must radiate at the real T.
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
        T_eff = T_c
        if T_cap_K > 0.0:
            T_eff = ti.min(T_c, T_cap_K)
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
def apply_evaporative_enthalpy_sink(
    H: ti.template(),
    T: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    cp_rho: ti.template(),
    use_cp_field: ti.i32,
    cp_rho_scalar: ti.f32,
    T_boil_K: ti.f32,
    T_onset_K: ti.f32,
    L_vapor_J_kg: ti.f32,
    R_spec_J_kgK: ti.f32,
    P_ref_Pa: ti.f32,
    C_acc: ti.f32,
    scale: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    energy_J_buf: ti.template(),
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Evaporative cooling (energy sink) on free-surface metal AND hot liquid.

    Surface weighting piles arc energy into bulk liquid under the torch; a
    free-surface-only sink never sees those cells, so the vapor-cap stays
    saturated. Liquid cells with T > onset are therefore included.

    Mass flux ~ Hertz–Knudsen / Anisimov:
        ṁ = C_acc · P_sat(T) / √(2 π R_spec T)
        q = ṁ · L_vapor · scale
    """
    eps = 1e-6
    energy_J_buf[None] = 0.0
    dx3 = dx * dx * dx
    two_pi = 2.0 * 3.141592653589793

    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        Tc = T[i, j, k]
        if Tc <= T_onset_K:
            continue

        has_gas = 0
        if i > 0 and flags[i - 1, j, k] == FLAG_GAS:
            has_gas = 1
        if i < nx - 1 and flags[i + 1, j, k] == FLAG_GAS:
            has_gas = 1
        if j > 0 and flags[i, j - 1, k] == FLAG_GAS:
            has_gas = 1
        if j < ny - 1 and flags[i, j + 1, k] == FLAG_GAS:
            has_gas = 1
        if k > 0 and flags[i, j, k - 1] == FLAG_GAS:
            has_gas = 1
        if k < nz - 1 and flags[i, j, k + 1] == FLAG_GAS:
            has_gas = 1
        dphi = 0.0
        if k > 0 and k < nz - 1:
            dphi = ti.abs(phi[i, j, k + 1] - phi[i, j, k - 1])
        is_surface = (has_gas == 1) or (dphi >= 0.05) or (phi[i, j, k] < 0.9)
        is_hot_liquid = f_l[i, j, k] > 0.5
        if (not is_surface) and (not is_hot_liquid):
            continue

        ramp = 1.0
        if Tc < T_boil_K:
            ramp = (Tc - T_onset_K) / (T_boil_K - T_onset_K + eps)
            ramp = ti.max(0.0, ti.min(1.0, ramp))
            ramp = ramp * ramp

        exponent = (L_vapor_J_kg / (R_spec_J_kgK + eps)) * (
            1.0 / (T_boil_K + eps) - 1.0 / (Tc + eps)
        )
        exponent = ti.max(ti.min(exponent, 12.0), -8.0)
        P_sat = P_ref_Pa * ti.math.exp(exponent)
        denom = ti.sqrt(two_pi * R_spec_J_kgK * ti.max(Tc, 300.0)) + eps
        m_dot = C_acc * P_sat / denom
        q = m_dot * L_vapor_J_kg * scale * ramp
        dH = q * dt / (dx + eps)

        cp_r = cp_rho_scalar
        if use_cp_field == 1:
            cp_r = cp_rho[i, j, k]
        dH_max = cp_r * ti.max(0.0, Tc - T_onset_K)
        dH = ti.min(dH, dH_max)
        if dH > 0.0:
            H[i, j, k] -= dH
            ti.atomic_add(energy_J_buf[None], dH * dx3)


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
def clamp_enthalpy_ceiling_dual(
    H: ti.template(),
    flags: ti.template(),
    cp_rho_field: ti.template(),
    alloy_id: ti.template(),
    H_liq_w: ti.f32,
    T_liq_w: ti.f32,
    H_liq_p: ti.f32,
    T_liq_p: ti.f32,
    T_vapor_cap: ti.f32,
    FLAG_GAS: ti.i32,
):
    """Vapor-cap enthalpy using each cell's composition-weighted H_liq / T_liquidus."""
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        cp_r = cp_rho_field[i, j, k]
        H_liq = _alloy_pick(alloy_id[i, j, k], H_liq_w, H_liq_p)
        T_liq = _alloy_pick(alloy_id[i, j, k], T_liq_w, T_liq_p)
        h_cap_liquid = H_liq + cp_r * (T_vapor_cap - T_liq)
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
    """Per-cell (T − T_prev)/dt [K/s]; positive while heating.

    Single-step |ΔT| above ``max_jump_K`` is treated as a discrete event
    (droplet inject, freeze clamp, vapor reset, gas→metal birth) — rate is
    zeroed for that step so telemetry/VTK are not polluted by 10⁶ K/s spikes.
    Gas cells keep ``T_prev`` synced so a newly created metal cell does not
    inherit a stale ambient/hot mismatch on the following step.
    """
    inv_dt = 1.0 / (dt_phys + 1e-12)
    # ~25 K/step at dt≈50 µs ⇒ 5e5 K/s. Real GMAW HAZ cool is typically
    # 10–few×10³ K/s; anything needing a bigger jump is not continuum cooling.
    max_jump_K = 25.0
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            dT_dt[i, j, k] = 0.0
            T_prev[i, j, k] = T[i, j, k]
            continue
        dT = T[i, j, k] - T_prev[i, j, k]
        if ti.abs(dT) > max_jump_K:
            dT_dt[i, j, k] = 0.0
        else:
            dT_dt[i, j, k] = dT * inv_dt
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
    T_cap_K:   ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Convective/radiative loss on gas-exposed metal (fluid AND solid).

    See apply_thermal_boundary_losses_variable for the flux-to-enthalpy
    conversion note (no ρ·cp factor — q·dt/dx is already J/m³).
    T_cap_K is an f32 safety ceiling (typically T_vapor_cap), not 2500 K.
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
        T_eff = T_c
        if T_cap_K > 0.0:
            T_eff = ti.min(T_c, T_cap_K)
        q_loss = 0.0
        if enable_conv == 1:
            q_loss += h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_loss += eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)

        H[i, j, k] -= q_loss * ti.f32(n_exposed) * dt / dx


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
def update_time_above_T_dual(
    T: ti.template(),
    flags: ti.template(),
    alloy_id: ti.template(),
    time_above_800: ti.template(),
    time_above_1100: ti.template(),
    time_above_solidus: ti.template(),
    dt: ti.f32,
    T_800: ti.f32,
    T_1100: ti.f32,
    T_sol_w: ti.f32,
    T_sol_p: ti.f32,
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
        T_sol = _alloy_pick(alloy_id[i, j, k], T_sol_w, T_sol_p)
        if t_c >= T_sol:
            time_above_solidus[i, j, k] += dt


