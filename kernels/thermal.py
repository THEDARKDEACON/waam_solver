"""Taichi kernels — heat injection, enthalpy, losses, phase."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    _alloy_pick,
    lookup_table_1d,
    temperature_from_enthalpy,
    temperature_from_enthalpy_mix,
)
from .grid_init import arc_deposition_weight

# ──────────────────────────────────────────────────────────────────────────────
#  KERNEL 2 — Gaussian Arc Heat Injection
# ──────────────────────────────────────────────────────────────────────────────
@ti.kernel
def inject_arc_heat_weights(
    flags: ti.template(),
    phi:   ti.template(),
    f_l:   ti.template(),
    scratch: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    sigma: ti.f32,
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
    FLAG_GAS: ti.i32,
    fs_iface: ti.i32,
):
    """Write Gaussian×deposition weights into ``scratch`` (C-order reduce later)."""
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for i, j, k in scratch:
        scratch[i, j, k] = 0.0
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
        scratch[i, j, k] = (
            ti.math.exp(-r2 * inv2s2) * w_dep
            * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
        )


@ti.kernel
def inject_arc_heat_apply(
    H:     ti.template(),
    flags: ti.template(),
    phi:   ti.template(),
    f_l:   ti.template(),
    norm_buf: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    Q_w:   ti.f32,
    sigma: ti.f32,
    dt:    ti.f32,
    dx3:   ti.f32,
    eta:   ti.f32,
    penetration_cells: ti.f32,
    enable_surface_weight: ti.i32,
    FLAG_GAS: ti.i32,
    fs_iface: ti.i32,
):
    """Deposit ΔH from a precomputed ordered weight sum in ``norm_buf``."""
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
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


def inject_arc_heat(
    H, flags, phi, f_l, norm_buf,
    arc_i, arc_j, arc_k, Q_w, sigma, dt, dx3, eta,
    penetration_cells, enable_surface_weight, FLAG_SOLID, FLAG_GAS, fs_iface,
    *, scratch=None, grid=None,
):
    """Gaussian arc heat with an index-ordered weight sum.

    Call with ``grid=`` (preferred) so the ordered reducer is used. Legacy
    positional args match the old kernel; ``FLAG_SOLID`` is unused.
    """
    if grid is None:
        raise TypeError("inject_arc_heat requires grid= for ordered reduction")
    from .reduce import ordered_sum_scratch, copy_total_to_f32
    scratch = scratch if scratch is not None else grid.reduce_scratch
    inject_arc_heat_weights(
        flags, phi, f_l, scratch,
        arc_i, arc_j, arc_k, sigma,
        penetration_cells, enable_surface_weight, FLAG_GAS, fs_iface,
    )
    ordered_sum_scratch(grid)
    copy_total_to_f32(grid.reduce_total, norm_buf)
    inject_arc_heat_apply(
        H, flags, phi, f_l, norm_buf,
        arc_i, arc_j, arc_k, Q_w, sigma, dt, dx3, eta,
        penetration_cells, enable_surface_weight, FLAG_GAS, fs_iface,
    )


@ti.func
def _goldak_arg(
    di: ti.f32,
    dj: ti.f32,
    dk: ti.f32,
    dir_x: ti.f32,
    dir_y: ti.f32,
    a_front: ti.f32,
    a_rear: ti.f32,
    b_axis: ti.f32,
    c_axis: ti.f32,
) -> ti.f32:
    """Goldak ARG = 3(x²/a² + y²/b² + z²/c²) in the travel frame (cell units)."""
    eps = 1e-6
    x_travel = di * dir_x + dj * dir_y
    y_trans = -di * dir_y + dj * dir_x
    is_front = x_travel >= 0.0
    a_axis = ti.select(is_front, a_front, a_rear)
    return 3.0 * (
        (x_travel * x_travel) / (a_axis * a_axis + eps)
        + (y_trans * y_trans) / (b_axis * b_axis + eps)
        + (dk * dk) / (c_axis * c_axis + eps)
    )


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

        [f_{f,r} / (a_{f,r} b c)] exp(-3 x²/a_{f,r}² - 3 y²/b² - 3 z²/c²)

    The shared 6√3/(π√π) factor is absorbed by energy renormalization when
    normalize_energy is on. With absolute CF mode the caller multiplies by CF.
    The 1/(a b c) factor is not optional: a_front and a_rear differ, so one
    global scale cannot represent both ellipsoids. With f_f + f_r = 2 the rear
    then carries f_r/2 of the plate power (0.7 for 0.6/1.4).
    Travel frame: (dir_x, dir_y) is the unit travel direction in the XY plane
    (from torch path). x_travel ≥ 0 ⇒ front ellipsoid.
    """
    eps = 1e-6
    x_travel = di * dir_x + dj * dir_y
    y_trans = -di * dir_y + dj * dir_x
    is_front = x_travel >= 0.0
    a_axis = ti.select(is_front, a_front, a_rear)
    frac = ti.select(is_front, ff, fr)
    abc = a_axis * b_axis * c_axis
    arg = _goldak_arg(
        di, dj, dk, dir_x, dir_y, a_front, a_rear, b_axis, c_axis,
    )
    return (frac / (abc + eps)) * ti.math.exp(-arg)


@ti.kernel
def inject_goldak_heat_weights(
    flags: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    scratch: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
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
    FLAG_GAS: ti.i32,
    fs_iface: ti.i32,
):
    """Write Goldak×deposition weights into ``scratch`` for an ordered sum."""
    for i, j, k in scratch:
        scratch[i, j, k] = 0.0
        if flags[i, j, k] == FLAG_GAS:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) + 0.5 - arc_i
        dj = ti.f32(j) + 0.5 - arc_j
        dk = ti.f32(k) + 0.5 - arc_k
        w = _goldak_pdf_weight(
            di, dj, dk, dir_x, dir_y, ff, fr,
            a_front, a_rear, b_axis, c_axis,
        )
        scratch[i, j, k] = (
            w * w_dep * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
        )


@ti.kernel
def inject_goldak_heat_apply(
    H: ti.template(),
    flags: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    norm_buf: ti.template(),
    scratch: ti.template(),
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
    FLAG_GAS: ti.i32,
    normalize_energy: ti.i32,
    fs_iface: ti.i32,
):
    """Deposit Goldak heat. Absolute-CF mode stages joules into ``scratch``."""
    w_total = norm_buf[None]
    energy_per_weight = 0.0
    if normalize_energy != 0 and w_total > 1e-9:
        energy_per_weight = Q_w * eta * dt / (w_total * dx3)
    Q_net = Q_w * eta
    CF = 6.0 * ti.sqrt(3.0) * Q_net / (ti.math.pi * ti.sqrt(ti.math.pi))
    for i, j, k in scratch:
        scratch[i, j, k] = 0.0
        if flags[i, j, k] == FLAG_GAS:
            continue
        if normalize_energy != 0 and energy_per_weight <= 0.0:
            continue
        w_dep = arc_deposition_weight(
            i, j, k, arc_i, arc_j, arc_k,
            f_l[i, j, k], penetration_cells, enable_surface_weight,
        )
        if w_dep < 1e-6:
            continue
        di = ti.f32(i) + 0.5 - arc_i
        dj = ti.f32(j) + 0.5 - arc_j
        dk = ti.f32(k) + 0.5 - arc_k
        if normalize_energy == 0:
            arg = _goldak_arg(
                di, dj, dk, dir_x, dir_y, a_front, a_rear, b_axis, c_axis,
            )
            if arg >= 9.0:
                continue
        w = _goldak_pdf_weight(
            di, dj, dk, dir_x, dir_y, ff, fr,
            a_front, a_rear, b_axis, c_axis,
        )
        if normalize_energy != 0:
            H[i, j, k] += energy_per_weight * w * w_dep
        else:
            dH = CF * w * w_dep * dt / dx3
            H[i, j, k] += dH
            scratch[i, j, k] = dH * dx3 * _fill_w(
                phi, flags, i, j, k, FLAG_GAS, fs_iface
            )


def inject_goldak_heat(
    H, flags, phi, f_l, norm_buf,
    arc_i, arc_j, arc_k, Q_w, dt, dx3, eta,
    dir_x, dir_y, ff, fr, a_front, a_rear, b_axis, c_axis,
    penetration_cells, enable_surface_weight,
    FLAG_SOLID, FLAG_GAS, normalize_energy, fs_iface,
    *, scratch=None, grid=None,
):
    """Goldak heat with an index-ordered weight (or ledger) sum.

    Requires ``grid=`` so the ordered reducer can run between weight and apply.
    """
    if grid is None:
        raise TypeError("inject_goldak_heat requires grid= for ordered reduction")
    from .reduce import ordered_sum_scratch, copy_total_to_f32
    scratch = scratch if scratch is not None else grid.reduce_scratch
    if int(normalize_energy) != 0:
        inject_goldak_heat_weights(
            flags, phi, f_l, scratch,
            arc_i, arc_j, arc_k, dir_x, dir_y, ff, fr,
            a_front, a_rear, b_axis, c_axis,
            penetration_cells, enable_surface_weight, FLAG_GAS, fs_iface,
        )
        ordered_sum_scratch(grid)
        copy_total_to_f32(grid.reduce_total, norm_buf)
    else:
        norm_buf[None] = 0.0
    inject_goldak_heat_apply(
        H, flags, phi, f_l, norm_buf, scratch,
        arc_i, arc_j, arc_k, Q_w, dt, dx3, eta,
        dir_x, dir_y, ff, fr, a_front, a_rear, b_axis, c_axis,
        penetration_cells, enable_surface_weight, FLAG_GAS,
        normalize_energy, fs_iface,
    )
    if int(normalize_energy) == 0:
        ordered_sum_scratch(grid)
        copy_total_to_f32(grid.reduce_total, norm_buf)

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
    # M9: include mushy-range sensible heat so H_liq = cp·T_liq + L
    # (previously H_liq = cp·T_sol + L dropped ~cp·ΔT_mush ≈ 3 % of L).
    H_sol = cp_rho * T_solidus
    H_liq = cp_rho * T_liquidus + L_rho
    dH_mush = H_liq - H_sol + 1e-9

    for i, j, k in H:
        h = H[i, j, k]
        if h <= H_sol:
            f_l[i, j, k] = 0.0
            T[i, j, k]   = h / cp_rho
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
            T[i, j, k]   = T_liquidus + (h - H_liq) / cp_rho
        else:
            fl = (h - H_sol) / dH_mush
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


@ti.func
def _advecting(flags: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32,
               FLAG_SOLID: ti.i32, FLAG_GAS: ti.i32, skip: ti.i32) -> ti.i32:
    f = flags[i, j, k]
    return ti.cast(f != FLAG_GAS and f != FLAG_SOLID and f != skip, ti.i32)


@ti.func
def _fill_w(phi: ti.template(), flags: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32,
            FLAG_GAS: ti.i32, fs_iface: ti.i32) -> ti.f32:
    """Metal fraction w of a cell; its thermal energy is w·H·Δx³.

    Every metal cell is full (w = 1) unless the free-surface LBM is on
    (fs_iface ≥ 0), where w is the clamped fill φ. Gas is 0.
    """
    w = 1.0
    if flags[i, j, k] == FLAG_GAS:
        w = 0.0
    elif fs_iface >= 0:
        w = ti.max(0.0, ti.min(1.0, phi[i, j, k]))
    return w


# A partly filled surface cell still exposes its whole face: surface sinks
# remove q·A·dt regardless of fill, i.e. ΔH = q·A·dt / (w·Δx³). Below this
# fill the divisor is held so thin cells cannot be driven to the floor.
_LOSS_MIN_FILL = 0.25


@ti.func
def _face_w(phi: ti.template(), flags: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32,
            w_c: ti.f32, FLAG_GAS: ti.i32, fs_iface: ti.i32,
            nx: ti.i32, ny: ti.i32, nz: ti.i32) -> ti.f32:
    """Face factor min(w_c, w_n)/w_c toward neighbour (i,j,k).

    Conductance scales with the wetted face fraction min(w_c, w_n) and the
    cell's heat capacity with w_c, so exchanged energy is antisymmetric and
    the factor never exceeds 1 (no tighter stability limit).
    """
    r = 1.0
    if fs_iface >= 0:
        ii = ti.max(0, ti.min(i, nx - 1))
        jj = ti.max(0, ti.min(j, ny - 1))
        kk = ti.max(0, ti.min(k, nz - 1))
        w_n = _fill_w(phi, flags, ii, jj, kk, FLAG_GAS, fs_iface)
        r = ti.min(w_c, w_n) / ti.max(w_c, 1e-6)
    return r


@ti.func
def _face_flux(
    H: ti.template(),
    vel: ti.template(),
    flags: ti.template(),
    li: ti.i32, lj: ti.i32, lk: ti.i32,
    di: ti.template(), dj: ti.template(), dk: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    skip: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
) -> ti.f32:
    """Enthalpy flux u_f·H_f through the face from cell L to its +d neighbour R.

    Zero unless both cells advect (faces on gas, solid, cells flagged ``skip``
    or the domain edge are no-penetration walls). H_f is the minmod-limited upwind (MUSCL) value.
    Both cells sharing a face evaluate this with the same arguments, so what
    leaves one cell enters the other exactly.
    """
    ri = li + di
    rj = lj + dj
    rk = lk + dk
    out = 0.0
    if (
        li >= 0 and lj >= 0 and lk >= 0
        and ri < nx and rj < ny and rk < nz
    ):
        if (
            _advecting(flags, li, lj, lk, FLAG_SOLID, FLAG_GAS, skip) == 1
            and _advecting(flags, ri, rj, rk, FLAG_SOLID, FLAG_GAS, skip) == 1
        ):
            uf = 0.5 * (vel[li, lj, lk] + vel[ri, rj, rk])
            HL = H[li, lj, lk]
            HR = H[ri, rj, rk]
            face = 0.0
            if uf > 0.0:
                HLL = _masked_at(H, flags, li - di, lj - dj, lk - dk, HL, FLAG_GAS, nx, ny, nz)
                face = HL + 0.5 * _minmod(HL - HLL, HR - HL)
            else:
                HRR = _masked_at(H, flags, ri + di, rj + dj, rk + dk, HR, FLAG_GAS, nx, ny, nz)
                face = HR - 0.5 * _minmod(HR - HL, HRR - HR)
            out = uf * face
    return out


@ti.func
def _advection_dH(
    H: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    i: ti.i32, j: ti.i32, k: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    skip: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
) -> ti.f32:
    """-div(u H) at cell (i,j,k) per unit time, in flux (conservative) form."""
    net = _face_flux(H, ux, flags, i, j, k, 1, 0, 0, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    net -= _face_flux(H, ux, flags, i - 1, j, k, 1, 0, 0, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    net += _face_flux(H, uy, flags, i, j, k, 0, 1, 0, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    net -= _face_flux(H, uy, flags, i, j - 1, k, 0, 1, 0, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    net += _face_flux(H, uz, flags, i, j, k, 0, 0, 1, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    net -= _face_flux(H, uz, flags, i, j, k - 1, 0, 0, 1, FLAG_SOLID, FLAG_GAS, skip, nx, ny, nz)
    return -net


@ti.kernel
def advect_diffuse_temperature(
    H:    ti.template(),    # Updated in-place (destination)
    H_src: ti.template(),   # M1: neighbour reads (pre-step snapshot)
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
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """
    Solve ∂H/∂t + ∇·(uH) = ∇·(k∇T) in lattice units.

    Under the free-surface LBM (fs_iface = FLAG_IFACE) conduction is
    fill-weighted (see ``_face_w``) and interface cells are left out of
    advection: their enthalpy moves with the LBM mass exchange instead.

    - Advection acts on H (the conserved variable): latent heat is transported
      with the melt, not just sensible heat.
    - SOLID cells conduct (no advection) — the substrate/frozen bead is a real
      heat sink instead of an insulator.
    - GAS neighbours are adiabatic: previously they acted as fixed ambient-T
      conductors, an unphysical heat sink at the free surface.
    - Advection is in flux form with minmod-limited 2nd-order upwind face
      values, so it only moves enthalpy between metal cells. The previous
      u·∇H form leaked energy wherever div(u) != 0 (free surface, Mach clamp).
    """
    for i, j, k in H:
        flag = flags[i, j, k]
        if flag == FLAG_GAS:
            continue

        T_c = T[i, j, k]
        H_c = H_src[i, j, k]

        # --- Diffusion: central differences on T, gas faces adiabatic ---
        T_im = _masked_at(T, flags, i - 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_ip = _masked_at(T, flags, i + 1, j, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jm = _masked_at(T, flags, i, j - 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_jp = _masked_at(T, flags, i, j + 1, k, T_c, FLAG_GAS, nx, ny, nz)
        T_km = _masked_at(T, flags, i, j, k - 1, T_c, FLAG_GAS, nx, ny, nz)
        T_kp = _masked_at(T, flags, i, j, k + 1, T_c, FLAG_GAS, nx, ny, nz)
        lap = T_ip + T_im + T_jp + T_jm + T_kp + T_km - 6.0 * T_c
        if fs_iface >= 0:
            w_c = _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
            lap = (
                _face_w(phi, flags, i - 1, j, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_im - T_c)
                + _face_w(phi, flags, i + 1, j, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_ip - T_c)
                + _face_w(phi, flags, i, j - 1, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_jm - T_c)
                + _face_w(phi, flags, i, j + 1, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_jp - T_c)
                + _face_w(phi, flags, i, j, k - 1, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_km - T_c)
                + _face_w(phi, flags, i, j, k + 1, w_c, FLAG_GAS, fs_iface, nx, ny, nz) * (T_kp - T_c)
            )

        dH = cp_rho * alpha_lu * lap * dt

        if flag != FLAG_SOLID:
            dH += _advection_dH(
                H_src, ux, uy, uz, flags, i, j, k, FLAG_SOLID, FLAG_GAS, fs_iface, nx, ny, nz,
            ) * dt

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
        if flags[i, j, k] == FLAG_GAS:
            cp_rho_field[i, j, k] = cp_rho_ref
            alpha_lu_field[i, j, k] = alpha_lu_ref
            dgamma_lu_field[i, j, k] = dgamma_lu_ref
            tau_field[i, j, k] = tau_ref
            continue
        T_c = T[i, j, k]
        if use_tables == 1:
            cp = lookup_table_1d(T_c, cp_T, cp_V, n_cp[None], cp_fallback)
            k_val = lookup_table_1d(T_c, k_T, k_V, n_k[None], k_fallback)
            cp_r = rho * cp
            cp_rho_field[i, j, k] = cp_r
            alpha_lu_field[i, j, k] = k_val / (cp_r + 1e-9) * dt / (dx * dx)
            # Solid cells conduct with k(T), cp(T); flow properties stay at the reference.
            if flags[i, j, k] == FLAG_SOLID:
                dgamma_lu_field[i, j, k] = dgamma_lu_ref
                tau_field[i, j, k] = tau_ref
            else:
                mu_val = lookup_table_1d(T_c, mu_T, mu_V, n_mu[None], mu_fallback)
                dgamma = lookup_table_1d(T_c, dgamma_T, dgamma_V, n_dgamma[None], dgamma_fallback)
                nu_lu = mu_val / (rho + 1e-9) * dt / (dx * dx)
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


@ti.func
def _conduction_exchange(
    T: ti.template(),
    flags: ti.template(),
    cp_rho_field: ti.template(),
    alpha_lu_field: ti.template(),
    i: ti.i32, j: ti.i32, k: ti.i32,
    T_c: ti.f32,
    k_c: ti.f32,
    FLAG_GAS: ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
) -> ti.f32:
    """Face exchange k_f·(T_n − T_c), k_f = mean of both cells' k·dt/dx² [J/(m³·K)].

    Out-of-domain and gas neighbours are adiabatic. The same k_f is used from
    both sides of the face, so conduction moves heat without creating it.
    """
    out = 0.0
    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
        if flags[i, j, k] != FLAG_GAS:
            k_n = cp_rho_field[i, j, k] * alpha_lu_field[i, j, k]
            out = 0.5 * (k_c + k_n) * (T[i, j, k] - T_c)
    return out


@ti.kernel
def advect_diffuse_temperature_variable(
    H: ti.template(),
    H_src: ti.template(),
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
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Variable-property advect_diffuse_temperature: ∇·(k∇T) with face-mean k.

    k(T_c)·∇²T (the previous form) drops k'(T)|∇T|² and, with k falling as T
    rises, creates heat wherever it flows from hot to cold cells. Advection is
    the same flux form as advect_diffuse_temperature.
    """
    for i, j, k in H:
        flag = flags[i, j, k]
        if flag == FLAG_GAS:
            continue

        T_c = T[i, j, k]
        H_c = H_src[i, j, k]
        k_c = cp_rho_field[i, j, k] * alpha_lu_field[i, j, k]
        w_c = _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)

        exch = _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i - 1, j, k, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i - 1, j, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz)
        exch += _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i + 1, j, k, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i + 1, j, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz)
        exch += _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i, j - 1, k, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i, j - 1, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz)
        exch += _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i, j + 1, k, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i, j + 1, k, w_c, FLAG_GAS, fs_iface, nx, ny, nz)
        exch += _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i, j, k - 1, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i, j, k - 1, w_c, FLAG_GAS, fs_iface, nx, ny, nz)
        exch += _conduction_exchange(
            T, flags, cp_rho_field, alpha_lu_field, i, j, k + 1, T_c, k_c, FLAG_GAS, nx, ny, nz,
        ) * _face_w(phi, flags, i, j, k + 1, w_c, FLAG_GAS, fs_iface, nx, ny, nz)

        dH = exch * dt

        if flag != FLAG_SOLID:
            dH += _advection_dH(
                H_src, ux, uy, uz, flags, i, j, k, FLAG_SOLID, FLAG_GAS, fs_iface, nx, ny, nz,
            ) * dt

        H[i, j, k] += dH


@ti.kernel
def update_phase_variable_cp(
    H: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    L_rho: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    H_sol: ti.f32,
    H_liq: ti.f32,
    rho: ti.f32,
    cp_T: ti.template(),
    cp_V: ti.template(),
    cp_H: ti.template(),
    n_cp: ti.template(),
):
    """Invert rho*integral(cp) plus the latent jump between the anchors.

    Liquid fraction stays linear in enthalpy from H_sol to H_liq. Temperature
    is the inverse of that curve, including the solid below the solidus.
    """
    dH_mush = H_liq - H_sol + 1e-9
    n = n_cp[None]
    for i, j, k in H:
        h = H[i, j, k]
        T[i, j, k] = temperature_from_enthalpy(
            h, rho, L_rho, T_solidus, T_liquidus, cp_T, cp_V, cp_H, n,
        )
        if h <= H_sol:
            f_l[i, j, k] = 0.0
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
        else:
            f_l[i, j, k] = (h - H_sol) / dH_mush


@ti.kernel
def update_phase_dual(
    H: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    alloy_frac: ti.template(),
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
    rho_w: ti.f32,
    rho_p: ti.f32,
    w_cp_T: ti.template(),
    w_cp_V: ti.template(),
    w_cp_H: ti.template(),
    w_n_cp: ti.template(),
    p_cp_T: ti.template(),
    p_cp_V: ti.template(),
    p_cp_H: ti.template(),
    p_n_cp: ti.template(),
):
    """Invert each alloy's enthalpy curve. Mixtures use the volume-weighted sum."""
    nw = w_n_cp[None]
    np_ = p_n_cp[None]
    for i, j, k in H:
        h = H[i, j, k]
        frac = alloy_frac[i, j, k]
        H_sol = _alloy_pick(frac, H_sol_w, H_sol_p)
        H_liq = _alloy_pick(frac, H_liq_w, H_liq_p)
        T[i, j, k] = temperature_from_enthalpy_mix(
            h, frac,
            rho_w, L_rho_w, T_sol_w, T_liq_w, w_cp_T, w_cp_V, w_cp_H, nw,
            rho_p, L_rho_p, T_sol_p, T_liq_p, p_cp_T, p_cp_V, p_cp_H, np_,
        )
        if h <= H_sol:
            f_l[i, j, k] = 0.0
        elif h >= H_liq:
            f_l[i, j, k] = 1.0
        else:
            f_l[i, j, k] = (h - H_sol) / (H_liq - H_sol + 1e-9)


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
    energy_acc: ti.template(),
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Convective/radiative loss on gas-exposed metal (fluid AND solid).

    Removed joules accumulate in energy_acc[0] (convection) and [1] (radiation).

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
        if k - 1 >= 0 and flags[i, j, k - 1] == FLAG_GAS:
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
        q_conv = 0.0
        q_rad = 0.0
        if enable_conv == 1:
            q_conv = h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_rad = eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)
        w_c = _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
        w_eff = ti.select(fs_iface >= 0, ti.max(w_c, _LOSS_MIN_FILL), 1.0)
        H[i, j, k] -= (q_conv + q_rad) * ti.f32(n_exposed) * dt / (dx * w_eff)
        face_dt = ti.f32(n_exposed) * dx * dx * dt * w_c / w_eff
        energy_acc[0] += ti.cast(q_conv * face_dt, ti.f64)
        energy_acc[1] += ti.cast(q_rad * face_dt, ti.f64)


@ti.kernel
def clamp_enthalpy_floor(
    H: ti.template(),
    flags: ti.template(),
    alloy_frac: ti.template(),
    H_amb_w: ti.f32,
    H_amb_p: ti.f32,
    FLAG_GAS: ti.i32,
    dx: ti.f32,
    energy_acc: ti.template(),
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Floor metal at the ambient enthalpy rho*integral(cp), per alloy.

    Added joules accumulate in energy_acc[2].
    """
    dx3 = dx * dx * dx
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h_min = _alloy_pick(alloy_frac[i, j, k], H_amb_w, H_amb_p)
        h = H[i, j, k]
        if h < h_min:
            H[i, j, k] = h_min
            energy_acc[2] += ti.cast((h_min - h) * dx3 * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface), ti.f64)


@ti.kernel
def clamp_enthalpy_floor_scalar(
    H: ti.template(),
    cp_rho: ti.f32,
    flags: ti.template(),
    T_floor: ti.f32,
    FLAG_GAS: ti.i32,
    dx: ti.f32,
    energy_acc: ti.template(),
    phi: ti.template(),
    fs_iface: ti.i32,
):
    h_min = cp_rho * T_floor
    dx3 = dx * dx * dx
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h = H[i, j, k]
        if h < h_min:
            H[i, j, k] = h_min
            energy_acc[2] += ti.cast((h_min - h) * dx3 * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface), ti.f64)


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
    fs_iface: ti.i32,
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
        # M3: evaporative sink on free-surface cells only (not bulk hot liquid).
        is_surface = (has_gas == 1) or (dphi >= 0.05) or (phi[i, j, k] < 0.9)
        if not is_surface:
            continue
        _ = f_l[i, j, k]  # kept for API / future surface-liquid gating

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
        w_c = _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
        w_eff = ti.select(fs_iface >= 0, ti.max(w_c, _LOSS_MIN_FILL), 1.0)
        dH = ti.min(dH / w_eff, dH_max)
        if dH > 0.0:
            H[i, j, k] -= dH
            ti.atomic_add(energy_J_buf[None], dH * dx3 * w_c)


@ti.kernel
def clamp_enthalpy_ceiling_scalar(
    H: ti.template(),
    flags: ti.template(),
    cp_rho: ti.f32,
    T_solidus: ti.f32,
    T_liquidus: ti.f32,
    T_vapor_cap: ti.f32,
    L_rho: ti.f32,
    dx: ti.f32,
    energy_J_buf: ti.template(),
    FLAG_GAS: ti.i32,
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Cap enthalpy so recovered T ≤ T_vapor_cap (matches update_phase).

    Phase recovery uses H_liq = cp·T_solidus + L (not cp·T_liquidus + L).
    Ceiling must be H_liq + cp·(T_cap − T_liquidus), otherwise T overshoots
    the vapor cap by ~(T_liquidus − T_solidus) ≈ 45 K and the HUD sticks at
    ~2977 °C whenever the hotspot saturates.

    Discarded energy is accumulated in ``energy_J_buf`` [J] via ΔH·Δx³.
    """
    H_sol = cp_rho * T_solidus
    H_liq = cp_rho * T_liquidus + L_rho  # M9-consistent
    h_cap_liquid = H_liq + cp_rho * (T_vapor_cap - T_liquidus)
    h_cap_solid = cp_rho * T_vapor_cap
    dx3 = dx * dx * dx
    energy_J_buf[None] = 0.0
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h = H[i, j, k]
        h_new = h
        if h > H_liq:
            if h > h_cap_liquid:
                h_new = h_cap_liquid
        else:
            if h > h_cap_solid:
                h_new = h_cap_solid
        if h_new < h:
            H[i, j, k] = h_new
            ti.atomic_add(energy_J_buf[None], (h - h_new) * dx3 * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface))


@ti.kernel
def clamp_enthalpy_ceiling_variable_cp(
    H: ti.template(),
    flags: ti.template(),
    H_cap: ti.f32,
    dx: ti.f32,
    energy_J_buf: ti.template(),
    FLAG_GAS: ti.i32,
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Cap H at the enthalpy of T_vapor_cap on the integral curve."""
    dx3 = dx * dx * dx
    energy_J_buf[None] = 0.0
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h = H[i, j, k]
        if h > H_cap:
            H[i, j, k] = H_cap
            ti.atomic_add(energy_J_buf[None], (h - H_cap) * dx3 * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface))


@ti.kernel
def clamp_enthalpy_ceiling_dual(
    H: ti.template(),
    flags: ti.template(),
    alloy_frac: ti.template(),
    H_cap_w: ti.f32,
    H_cap_p: ti.f32,
    dx: ti.f32,
    energy_J_buf: ti.template(),
    FLAG_GAS: ti.i32,
    phi: ti.template(),
    fs_iface: ti.i32,
):
    """Cap each cell at its alloy's enthalpy of T_vapor_cap."""
    dx3 = dx * dx * dx
    energy_J_buf[None] = 0.0
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        h_cap = _alloy_pick(alloy_frac[i, j, k], H_cap_w, H_cap_p)
        h = H[i, j, k]
        if h > h_cap:
            H[i, j, k] = h_cap
            ti.atomic_add(energy_J_buf[None], (h - h_cap) * dx3 * _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface))


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
    max_jump_K = ti.max(25.0, 5.0e5 * dt_phys)
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
    energy_acc: ti.template(),
    phi: ti.template(),
    fs_iface: ti.i32,
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
        if k - 1 >= 0 and flags[i, j, k - 1] == FLAG_GAS:
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
        q_conv = 0.0
        q_rad = 0.0
        if enable_conv == 1:
            q_conv = h_conv * (T_eff - T_amb)
        if enable_rad == 1:
            q_rad = eps_rad * sigma_sb * (T_eff ** 4 - T_amb ** 4)
        w_c = _fill_w(phi, flags, i, j, k, FLAG_GAS, fs_iface)
        w_eff = ti.select(fs_iface >= 0, ti.max(w_c, _LOSS_MIN_FILL), 1.0)
        H[i, j, k] -= (q_conv + q_rad) * ti.f32(n_exposed) * dt / (dx * w_eff)
        face_dt = ti.f32(n_exposed) * dx * dx * dt * w_c / w_eff
        energy_acc[0] += ti.cast(q_conv * face_dt, ti.f64)
        energy_acc[1] += ti.cast(q_rad * face_dt, ti.f64)


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


