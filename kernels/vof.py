"""Taichi kernels — VOF, wetting, CSF, solidify/remelt."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    _alloy_pick,
)

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
    """Derive FLUID / GAS / IFACE from φ and liquid fraction.

    Substrate band (k < nz_solid): keep existing metal as SOLID, but never
    convert GAS → SOLID. A previous blanket assignment filled the whole
    domain footprint with invented plate metal beside partial coupons.
    """
    for i, j, k in phi:
        if k < nz_solid:
            if flags[i, j, k] != FLAG_GAS:
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
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    H_sol: ti.f32,
    T_solidus: ti.f32,
    zero_velocity: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Promote solidified weld metal to static SOLID (bead growth / interpass).

    Clamps H ≤ H_sol so f_l/T recovered next step stay consistent with a
    fully solid cell (avoids freeze→remelt chatter from leftover latent H).
    """
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if phi[i, j, k] < 0.85:
            continue
        if f_l[i, j, k] < 0.02 and T[i, j, k] < T_solidus:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 0.0
            if H[i, j, k] > H_sol:
                H[i, j, k] = H_sol
            if zero_velocity == 1:
                ux[i, j, k] = 0.0
                uy[i, j, k] = 0.0
                uz[i, j, k] = 0.0


@ti.kernel
def solidify_cooled_metal_dual(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    alloy_id: ti.template(),
    H_sol_w: ti.f32,
    T_sol_w: ti.f32,
    H_sol_p: ti.f32,
    T_sol_p: ti.f32,
    zero_velocity: ti.i32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Birth-alloy freeze: plate cells use plate T_solidus / H_sol."""
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if phi[i, j, k] < 0.85:
            continue
        T_sol = _alloy_pick(alloy_id[i, j, k], T_sol_w, T_sol_p)
        H_sol = _alloy_pick(alloy_id[i, j, k], H_sol_w, H_sol_p)
        if f_l[i, j, k] < 0.02 and T[i, j, k] < T_sol:
            flags[i, j, k] = FLAG_SOLID
            phi[i, j, k] = 1.0
            f_l[i, j, k] = 0.0
            if H[i, j, k] > H_sol:
                H[i, j, k] = H_sol
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
    L_rho: ti.f32,
    H_sol: ti.f32,
    H_liq: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Re-open SOLID → FLUID when enthalpy already indicates melting.

    Driven by H (not T). Previously remelt reconstructed H = cp·T + f_l·L
    from temperature, which *created* latent heat whenever a sensible-only
    solid crossed T_solidus — prolonging liquid lifetime and, with VOF flag
    fill-in, runaway heating toward the vapor cap.
    """
    margin = 0.02 * L_rho
    for i, j, k in T:
        if flags[i, j, k] != FLAG_SOLID:
            continue
        h = H[i, j, k]
        if h <= H_sol + margin:
            continue
        if h >= H_liq:
            f_l[i, j, k] = 1.0
        else:
            f_l[i, j, k] = (h - H_sol) / (L_rho + 1e-9)
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_FLUID

        # H unchanged — latent heat must arrive via conduction/advection.


@ti.kernel
def remelt_hot_solid_dual(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    alloy_id: ti.template(),
    L_rho_w: ti.f32,
    H_sol_w: ti.f32,
    H_liq_w: ti.f32,
    L_rho_p: ti.f32,
    H_sol_p: ti.f32,
    H_liq_p: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Enthalpy-driven remelt with birth-alloy H_sol / L."""
    for i, j, k in T:
        if flags[i, j, k] != FLAG_SOLID:
            continue
        L_rho = _alloy_pick(alloy_id[i, j, k], L_rho_w, L_rho_p)
        H_sol = _alloy_pick(alloy_id[i, j, k], H_sol_w, H_sol_p)
        H_liq = _alloy_pick(alloy_id[i, j, k], H_liq_w, H_liq_p)
        margin = 0.02 * L_rho
        h = H[i, j, k]
        if h <= H_sol + margin:
            continue
        if h >= H_liq:
            f_l[i, j, k] = 1.0
        else:
            f_l[i, j, k] = (h - H_sol) / (L_rho + 1e-9)
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_FLUID


@ti.kernel
def remelt_hot_solid_scalar(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    L_rho: ti.f32,
    H_sol: ti.f32,
    H_liq: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
):
    """Scalar-cp twin of remelt_hot_solid (enthalpy-driven, no H rewrite)."""
    margin = 0.02 * L_rho
    for i, j, k in T:
        if flags[i, j, k] != FLAG_SOLID:
            continue
        h = H[i, j, k]
        if h <= H_sol + margin:
            continue
        if h >= H_liq:
            f_l[i, j, k] = 1.0
        else:
            f_l[i, j, k] = (h - H_sol) / (L_rho + 1e-9)
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
def solidify_trailing_pool_dual(
    T: ti.template(),
    H: ti.template(),
    f_l: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    cp_rho_field: ti.template(),
    alloy_id: ti.template(),
    arc_i: ti.f32,
    arc_j: ti.f32,
    arc_k: ti.f32,
    dir_x: ti.f32,
    dir_y: ti.f32,
    dir_z: ti.f32,
    lookback_cells: ti.f32,
    T_freeze_w: ti.f32,
    T_sol_w: ti.f32,
    T_freeze_p: ti.f32,
    T_sol_p: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Trailing freeze using each cell's birth-alloy solidus / freeze temperature."""
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
        T_freeze = _alloy_pick(alloy_id[i, j, k], T_freeze_w, T_freeze_p)
        T_sol = _alloy_pick(alloy_id[i, j, k], T_sol_w, T_sol_p)
        if T[i, j, k] >= T_freeze:
            continue
        cp_r = cp_rho_field[i, j, k]
        H[i, j, k] = cp_r * (T_sol - 1.0)
        T[i, j, k] = T_sol - 1.0
        f_l[i, j, k] = 0.0
        phi[i, j, k] = 1.0
        flags[i, j, k] = FLAG_SOLID
        ux[i, j, k] = 0.0
        uy[i, j, k] = 0.0
        uz[i, j, k] = 0.0


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


