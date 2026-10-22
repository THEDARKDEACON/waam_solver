"""
Körner / Thürey free-surface LBM.

Metal is a single-phase LBM fluid and gas cells carry no populations.
Interface cells (FLAG_IFACE) hold the fill fraction ε = m/ρ in ``phi``;
FLUID cells have ε = 1 and GAS cells ε = 0. After collision each step:

1. ``fs_prepare``            FLUID cells that touch GAS become IFACE.
2. ``fs_stream``             pull stream. Populations arriving from GAS are
                             rebuilt with the free-surface pressure condition
                             f_q = f^eq_q(ρ_G,u) + f^eq_q̄(ρ_G,u) − f*_q̄ with
                             ρ_G = 1 + 3γκ, and IFACE cells exchange mass
                             Δm = Σ_q s_q (f*_q(x − e_q) − f*_q̄(x)).
3. ``fs_mark_neighbors``     filled cells claim their GAS neighbours and veto
                             emptying of adjacent IFACE cells.
4. ``fs_init_new_iface``     claimed GAS cells take ρ, u, H, T, f_l from their
                             metal neighbours; FLUID cells beside an emptied
                             cell are marked.
5. ``fs_promote_new_iface``  marked cells become IFACE.
6. ``fs_distribute_excess``  filled/emptied cells hand their excess mass to
                             IFACE neighbours and become FLUID/GAS.
7. ``fs_update_phi``         ε = m/ρ.
8. ``fs_apply_energy``       H = E/ε.

Mass lives in ``phi`` between steps (m = ε·ρ is rebuilt in step 2), so the
deposition, remelt and solidify kernels that write ``phi`` and flags need no
knowledge of this module.

Thermal energy is fill-weighted, E = ε·H·Δx³ (``thermal._fill_w``). Enthalpy
crosses every face that touches an interface cell with the LBM mass flux,
taken from the upwind cell, so it follows the metal; new interface cells
start empty and carry no energy. Thermal advection skips those faces.
"""

from __future__ import annotations

from waam_twin.compiler import ti
from ..lattice import EX, EY, EZ, W, OPP
from .vof import _brackbill_curvature_at

MARK_NONE = 0
MARK_FILL = 1
MARK_EMPTY = 2
MARK_FROM_GAS = 3
MARK_FROM_FLUID = 4

# Excess fill beyond which an interface cell converts (fraction of ρ).
_CONVERT_EPS = 1e-3


@ti.func
def _inside(i, j, k, nx, ny, nz):
    return i >= 0 and j >= 0 and k >= 0 and i < nx and j < ny and k < nz


@ti.func
def _feq(q: ti.template(), rho, ux, uy, uz):
    eu = EX[q] * ux + EY[q] * uy + EZ[q] * uz
    u2 = ux * ux + uy * uy + uz * uz
    return W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)


@ti.func
def _touches_gas(flags, i, j, k, FLAG_GAS, nx, ny, nz):
    hit = False
    for q in ti.static(range(1, 7)):
        ni = i + EX[q]
        nj = j + EY[q]
        nk = k + EZ[q]
        if _inside(ni, nj, nk, nx, ny, nz):
            if flags[ni, nj, nk] == FLAG_GAS:
                hit = True
    return hit


@ti.kernel
def fs_prepare(
    flags: ti.template(),
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """FLUID never touches GAS: relabel such cells IFACE (ε stays in phi)."""
    for i, j, k in flags:
        if flags[i, j, k] == FLAG_FLUID:
            touch = 0
            for q in ti.static(range(1, 19)):
                ni = i + EX[q]
                nj = j + EY[q]
                nk = k + EZ[q]
                if _inside(ni, nj, nk, nx, ny, nz):
                    if flags[ni, nj, nk] == FLAG_GAS:
                        touch = 1
            if touch == 1:
                flags[i, j, k] = FLAG_IFACE


@ti.kernel
def fs_stream(
    f_post: ti.template(),
    f_new: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    mass: ti.template(),
    mark: ti.template(),
    H: ti.template(),
    energy: ti.template(),
    gamma_lu: ti.f32,
    kappa_max: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Pull stream with gas reconstruction and interface mass/energy exchange.

    ``f_post`` is the post-collision buffer, ``rho`` still holds the
    pre-stream density from collision, ``phi`` the pre-stream fill.
    ``energy`` receives ε·H plus the enthalpy carried across interface faces.
    """
    for i, j, k in flags:
        fl = flags[i, j, k]
        if fl == FLAG_SOLID or fl == FLAG_GAS:
            continue
        is_if = fl == FLAG_IFACE
        rho_g = 1.0
        if is_if and gamma_lu != 0.0:
            kappa, _g = _brackbill_curvature_at(
                phi, flags, i, j, k, FLAG_SOLID, nx, ny, nz, 1e-6,
            )
            kappa = ti.max(-kappa_max, ti.min(kappa_max, kappa))
            rho_g = 1.0 + 3.0 * gamma_lu * kappa
        u_x = ux[i, j, k]
        u_y = uy[i, j, k]
        u_z = uz[i, j, k]
        eps_c = phi[i, j, k]
        rho_old = rho[i, j, k]
        H_c = H[i, j, k]
        dm = 0.0
        dE = 0.0
        rho_new = 0.0
        n_gas = 0
        n_fluid = 0
        for q in ti.static(range(19)):
            ni = i - EX[q]
            nj = j - EY[q]
            nk = k - EZ[q]
            f_out = f_post[OPP[q], i, j, k]
            val = f_out
            if _inside(ni, nj, nk, nx, ny, nz):
                fn = flags[ni, nj, nk]
                if fn == FLAG_GAS:
                    n_gas += 1
                    val = (
                        _feq(q, rho_g, u_x, u_y, u_z)
                        + _feq(OPP[q], rho_g, u_x, u_y, u_z)
                        - f_out
                    )
                elif fn != FLAG_SOLID:
                    f_in = f_post[q, ni, nj, nk]
                    val = f_in
                    if fn != FLAG_IFACE:
                        n_fluid += 1
                    if is_if or fn == FLAG_IFACE:
                        s = 1.0
                        if is_if and fn == FLAG_IFACE:
                            s = 0.5 * (eps_c + phi[ni, nj, nk])
                        flux = s * (f_in - f_out)
                        if is_if:
                            dm += flux
                        h_up = H_c
                        if flux > 0.0:
                            h_up = H[ni, nj, nk]
                        dE += flux * h_up
            f_new[q, i, j, k] = val
            rho_new += val
        rho[i, j, k] = rho_new
        m = rho_new
        mk = MARK_NONE
        if is_if:
            m = eps_c * rho_old + dm
            # A cell with no gas neighbour is buried and becomes fluid whatever
            # its mass (stagnant liquid gives it no net inflow, so it would stay
            # a void); its deficit goes to the surface via fs_settle_unplaced. A
            # nearly empty cell with no fluid neighbour is empty (stray spray).
            if m > (1.0 + _CONVERT_EPS) * rho_new or (n_gas == 0 and n_fluid > 0):
                mk = MARK_FILL
            elif m < -_CONVERT_EPS * rho_new or (
                n_fluid == 0 and n_gas > 0 and m < 0.1 * rho_new
            ):
                mk = MARK_EMPTY
        mass[i, j, k] = m
        mark[i, j, k] = mk
        energy[i, j, k] = ti.max(0.0, ti.min(1.0, eps_c)) * H_c + dE


@ti.kernel
def fs_mark_neighbors(
    flags: ti.template(),
    mark: ti.template(),
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """GAS beside a filled cell → new interface; filling vetoes emptying."""
    for i, j, k in flags:
        is_gas = flags[i, j, k] == FLAG_GAS
        if is_gas or mark[i, j, k] == MARK_EMPTY:
            near_fill = 0
            for q in ti.static(range(1, 19)):
                ni = i + EX[q]
                nj = j + EY[q]
                nk = k + EZ[q]
                if _inside(ni, nj, nk, nx, ny, nz):
                    if mark[ni, nj, nk] == MARK_FILL:
                        near_fill = 1
            if near_fill == 1:
                if is_gas:
                    mark[i, j, k] = MARK_FROM_GAS
                else:
                    mark[i, j, k] = MARK_NONE


@ti.kernel
def fs_init_new_iface(
    f_src: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    mark: ti.template(),
    mass: ti.template(),
    energy: ti.template(),
    H: ti.template(),
    T: ti.template(),
    f_l: ti.template(),
    FLAG_FLUID: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Initialise claimed GAS cells from metal neighbours; mark FLUID beside emptied.

    The new cell starts with zero mass and energy at the neighbours' mean ρ,
    u and thermal state (H is intensive, so it only sets the temperature the
    first inflow sees).
    """
    for i, j, k in flags:
        mk = mark[i, j, k]
        if mk == MARK_FROM_GAS:
            n = 0
            rs = 0.0
            uxs = 0.0
            uys = 0.0
            uzs = 0.0
            hs = 0.0
            ts = 0.0
            ls = 0.0
            for q in ti.static(range(1, 19)):
                ni = i + EX[q]
                nj = j + EY[q]
                nk = k + EZ[q]
                if _inside(ni, nj, nk, nx, ny, nz):
                    fn = flags[ni, nj, nk]
                    if (fn == FLAG_FLUID or fn == FLAG_IFACE) and mark[ni, nj, nk] != MARK_FROM_GAS:
                        n += 1
                        rs += rho[ni, nj, nk]
                        uxs += ux[ni, nj, nk]
                        uys += uy[ni, nj, nk]
                        uzs += uz[ni, nj, nk]
                        hs += H[ni, nj, nk]
                        ts += T[ni, nj, nk]
                        ls += f_l[ni, nj, nk]
            inv = 1.0 / ti.max(1.0, ti.cast(n, ti.f32))
            r = 1.0
            if n > 0:
                r = rs * inv
            u_x = uxs * inv
            u_y = uys * inv
            u_z = uzs * inv
            for q in ti.static(range(19)):
                f_src[q, i, j, k] = _feq(q, r, u_x, u_y, u_z)
            rho[i, j, k] = r
            ux[i, j, k] = u_x
            uy[i, j, k] = u_y
            uz[i, j, k] = u_z
            mass[i, j, k] = 0.0
            energy[i, j, k] = 0.0
            if n > 0:
                H[i, j, k] = hs * inv
                T[i, j, k] = ts * inv
                f_l[i, j, k] = ls * inv
        elif flags[i, j, k] == FLAG_FLUID:
            near_empty = 0
            for q in ti.static(range(1, 19)):
                ni = i + EX[q]
                nj = j + EY[q]
                nk = k + EZ[q]
                if _inside(ni, nj, nk, nx, ny, nz):
                    if mark[ni, nj, nk] == MARK_EMPTY:
                        near_empty = 1
            if near_empty == 1:
                mark[i, j, k] = MARK_FROM_FLUID


@ti.kernel
def fs_promote_new_iface(
    flags: ti.template(),
    mark: ti.template(),
    mass: ti.template(),
    rho: ti.template(),
    FLAG_IFACE: ti.i32,
):
    for i, j, k in flags:
        mk = mark[i, j, k]
        if mk == MARK_FROM_GAS:
            flags[i, j, k] = FLAG_IFACE
        elif mk == MARK_FROM_FLUID:
            flags[i, j, k] = FLAG_IFACE
            mass[i, j, k] = rho[i, j, k]


@ti.kernel
def fs_distribute_excess(
    flags: ti.template(),
    mark: ti.template(),
    mass: ti.template(),
    rho: ti.template(),
    energy: ti.template(),
    f_l: ti.template(),
    dx: ti.f32,
    energy_acc: ti.template(),
    mass_lost: ti.template(),
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Filled → FLUID, emptied → GAS; excess mass split over IFACE neighbours.

    The excess carries its share of the cell's energy (all of it for an
    emptied cell). Mass with no interface neighbour to receive it accumulates
    (lattice units, signed) in ``mass_lost`` and its energy in energy_acc[3].
    """
    dx3 = dx * dx * dx
    for i, j, k in flags:
        mk = mark[i, j, k]
        if mk == MARK_FILL or mk == MARK_EMPTY:
            m = mass[i, j, k]
            r = rho[i, j, k]
            E = energy[i, j, k]
            ex = m
            ex_E = E
            if mk == MARK_FILL:
                ex = m - r
                ex_E = E * ex / ti.max(m, 1e-6)
            cnt = 0
            for q in ti.static(range(1, 19)):
                ni = i + EX[q]
                nj = j + EY[q]
                nk = k + EZ[q]
                if _inside(ni, nj, nk, nx, ny, nz):
                    mn = mark[ni, nj, nk]
                    if flags[ni, nj, nk] == FLAG_IFACE and mn != MARK_FILL and mn != MARK_EMPTY:
                        cnt += 1
            # A buried cell's deficit can be most of a cell; local neighbours
            # would be driven past the ε clamp, so it goes to the surface fix-up.
            if mk == MARK_FILL and ex < 0.0:
                cnt = 0
            if cnt > 0:
                inv = 1.0 / ti.cast(cnt, ti.f32)
                for q in ti.static(range(1, 19)):
                    ni = i + EX[q]
                    nj = j + EY[q]
                    nk = k + EZ[q]
                    if _inside(ni, nj, nk, nx, ny, nz):
                        mn = mark[ni, nj, nk]
                        if flags[ni, nj, nk] == FLAG_IFACE and mn != MARK_FILL and mn != MARK_EMPTY:
                            ti.atomic_add(mass[ni, nj, nk], ex * inv)
                            ti.atomic_add(energy[ni, nj, nk], ex_E * inv)
            else:
                mass_lost[None] += ti.cast(ex, ti.f64)
                energy_acc[3] -= ti.cast(ex_E * dx3, ti.f64)
            energy[i, j, k] = E - ex_E
            if mk == MARK_FILL:
                flags[i, j, k] = FLAG_FLUID
                mass[i, j, k] = r
            else:
                flags[i, j, k] = FLAG_GAS
                mass[i, j, k] = 0.0
                f_l[i, j, k] = 0.0


@ti.kernel
def fs_count_surface(
    flags: ti.template(),
    mass_lost: ti.template(),
    n_surface: ti.template(),
    share: ti.template(),
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Count surface IFACE cells and take the pending unplaced mass per cell."""
    n_surface[None] = 0
    for i, j, k in flags:
        if flags[i, j, k] == FLAG_IFACE and _touches_gas(flags, i, j, k, FLAG_GAS, nx, ny, nz):
            ti.atomic_add(n_surface[None], 1)
    share[None] = 0.0
    if n_surface[None] > 0:
        share[None] = ti.cast(mass_lost[None] / ti.cast(n_surface[None], ti.f64), ti.f32)
        mass_lost[None] = 0.0


@ti.kernel
def fs_settle_unplaced(
    flags: ti.template(),
    mass: ti.template(),
    energy: ti.template(),
    H: ti.template(),
    share: ti.template(),
    dx: ti.f32,
    energy_acc: ti.template(),
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Spread unplaced excess mass evenly over the free surface (global fix-up).

    Metal moves at each surface cell's own enthalpy; the energy is booked to
    energy_acc[3] so the ledger closes.
    """
    dm = share[None]
    dx3 = dx * dx * dx
    for i, j, k in flags:
        if dm != 0.0 and flags[i, j, k] == FLAG_IFACE and _touches_gas(
            flags, i, j, k, FLAG_GAS, nx, ny, nz
        ):
            mass[i, j, k] += dm
            dE = dm * H[i, j, k]
            energy[i, j, k] += dE
            energy_acc[3] += ti.cast(dE * dx3, ti.f64)


@ti.kernel
def fs_update_phi(
    flags: ti.template(),
    phi: ti.template(),
    mass: ti.template(),
    rho: ti.template(),
    mark: ti.template(),
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
):
    """ε = m/ρ on IFACE (kept slightly outside [0, 1] so mass is not clipped)."""
    for i, j, k in flags:
        fl = flags[i, j, k]
        if fl == FLAG_IFACE:
            r = rho[i, j, k]
            e = 0.0
            if r > 1e-6:
                e = mass[i, j, k] / r
            phi[i, j, k] = ti.max(-0.1, ti.min(1.1, e))
        elif fl == FLAG_FLUID:
            phi[i, j, k] = 1.0
        elif fl == FLAG_GAS:
            phi[i, j, k] = 0.0
        mark[i, j, k] = MARK_NONE


# Below this fill H = E/ε is ill-conditioned; such cells keep H and the
# (tiny) energy difference is booked.
_H_MIN_FILL = 0.05


@ti.kernel
def fs_apply_energy(
    H: ti.template(),
    flags: ti.template(),
    phi: ti.template(),
    energy: ti.template(),
    dx: ti.f32,
    energy_acc: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Recover the intensive H = E/ε on FLUID and IFACE cells."""
    dx3 = dx * dx * dx
    for i, j, k in flags:
        fl = flags[i, j, k]
        if fl == FLAG_GAS or fl == FLAG_SOLID:
            continue
        w = ti.max(0.0, ti.min(1.0, phi[i, j, k]))
        E = energy[i, j, k]
        if w >= _H_MIN_FILL:
            H[i, j, k] = E / w
        else:
            energy_acc[3] += ti.cast((w * H[i, j, k] - E) * dx3, ti.f64)


@ti.kernel
def fs_update_phase_flags(
    phi: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    f_src: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    nz_solid: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_IFACE: ti.i32,
):
    """Melt / freeze flags for the free-surface path; never creates or deletes gas.

    Same thresholds as ``update_flags_from_phi``. Remelted cells restart from
    quiescent equilibrium so they do not inherit stale wall populations; a
    partly filled one reopens as IFACE so its fill is kept.
    Unlike ``sync_phi_liquid_fraction`` nearly empty interface cells keep their
    f_l, otherwise Darcy drag would pin the free surface.
    """
    for i, j, k in phi:
        flag = flags[i, j, k]
        fl = f_l[i, j, k]
        if flag == FLAG_GAS:
            f_l[i, j, k] = 0.0
            continue
        melt = fl > 0.1
        freeze = fl < 0.05
        if k >= nz_solid:
            melt = melt and phi[i, j, k] > 0.55
            freeze = freeze and phi[i, j, k] > 0.85
        if flag == FLAG_SOLID:
            if melt:
                flags[i, j, k] = FLAG_FLUID
                if phi[i, j, k] < 0.999:
                    flags[i, j, k] = FLAG_IFACE
                rho[i, j, k] = 1.0
                ux[i, j, k] = 0.0
                uy[i, j, k] = 0.0
                uz[i, j, k] = 0.0
                for q in ti.static(range(19)):
                    f_src[q, i, j, k] = W[q]
            else:
                f_l[i, j, k] = 0.0
        elif freeze:
            flags[i, j, k] = FLAG_SOLID
            f_l[i, j, k] = 0.0


@ti.kernel
def fs_deposit_into_interface(
    phi: ti.template(),
    flags: ti.template(),
    H: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    footprint_r: ti.f32,
    droplet_radius: ti.f32,
    target_vol: ti.f32,
    H_drop: ti.f32,
    vol_acc: ti.template(),
    real_acc: ti.template(),
    cell_vol: ti.f32,
    FLAG_IFACE: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """Top up partly filled interface cells in the droplet footprint.

    Runs before ``feed_wire_surface`` on the same budget. Droplet metal placed
    only on GAS cells would bury the partly filled interface layer as
    permanent voids. A topped-up cell gains (1 − ε)·Δx³ of metal and its
    enthalpy mixes by fill, so the added energy is (1 − ε)·H_drop·Δx³.
    """
    search_r = ti.max(footprint_r, droplet_radius)
    i_min = ti.max(0, ti.cast(arc_i - search_r, ti.i32))
    i_max = ti.min(nx, ti.cast(arc_i + search_r, ti.i32) + 1)
    j_min = ti.max(0, ti.cast(arc_j - search_r, ti.i32))
    j_max = ti.min(ny, ti.cast(arc_j + search_r, ti.i32) + 1)
    k_lo = ti.max(0, ti.cast(arc_k - 1.0, ti.i32))
    k_hi = ti.min(nz, ti.cast(arc_k + ti.max(droplet_radius + 3.0, 4.0), ti.i32) + 1)
    foot2 = search_r * search_r * 1.44
    for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_lo, k_hi)):
        if flags[i, j, k] != FLAG_IFACE:
            continue
        di = ti.f32(i) - arc_i
        dj = ti.f32(j) - arc_j
        if di * di + dj * dj > foot2:
            continue
        e = ti.max(0.0, ti.min(1.0, phi[i, j, k]))
        need = (1.0 - e) * cell_vol
        if need <= 0.0 or vol_acc[None] >= target_vol:
            continue
        reserved = ti.atomic_add(vol_acc[None], need)
        if reserved + need > target_vol:
            ti.atomic_sub(vol_acc[None], need)
        else:
            H[i, j, k] = e * H[i, j, k] + (1.0 - e) * H_drop
            phi[i, j, k] = 1.0
            ti.atomic_add(real_acc[None], need)


@ti.kernel
def area_weighted_marangoni(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    T: ti.template(),
    phi: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    dgamma_dT_lu: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Marangoni with |∇φ| area weight so total force is mesh-independent.

    F = (dγ/dT) · (I − n⊗n) · ∇T · |∇φ|
    """
    for i, j, k in Fx:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        if f_l[i, j, k] < 0.1:
            continue
        dpx = 0.5 * (phi[ti.min(i + 1, nx - 1), j, k] - phi[ti.max(i - 1, 0), j, k])
        dpy = 0.5 * (phi[i, ti.min(j + 1, ny - 1), k] - phi[i, ti.max(j - 1, 0), k])
        dpz = 0.5 * (phi[i, j, ti.min(k + 1, nz - 1)] - phi[i, j, ti.max(k - 1, 0)])
        gmag = ti.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
        if gmag < 0.05:
            continue
        nx_n = dpx / gmag
        ny_n = dpy / gmag
        nz_n = dpz / gmag
        dTx = 0.5 * (T[ti.min(i + 1, nx - 1), j, k] - T[ti.max(i - 1, 0), j, k])
        dTy = 0.5 * (T[i, ti.min(j + 1, ny - 1), k] - T[i, ti.max(j - 1, 0), k])
        dTz = 0.5 * (T[i, j, ti.min(k + 1, nz - 1)] - T[i, j, ti.max(k - 1, 0)])
        # Project ∇T onto tangent plane
        tdot = dTx * nx_n + dTy * ny_n + dTz * nz_n
        tTx = dTx - tdot * nx_n
        tTy = dTy - tdot * ny_n
        tTz = dTz - tdot * nz_n
        scale = dgamma_dT_lu * gmag
        Fx[i, j, k] += scale * tTx
        Fy[i, j, k] += scale * tTy
        Fz[i, j, k] += scale * tTz
