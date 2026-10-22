"""Taichi kernels — electrostatics / J×B."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    _cell_rho,
)
from .weld import _sigma_at

@ti.kernel
def elec_build_sigma(
    sigma: ti.template(),
    f_l: ti.template(),
    flags: ti.template(),
    sigma_liquid: ti.f32,
    sigma_solid: ti.f32,
    sigma_gas: ti.f32,
    FLAG_GAS: ti.i32,
):
    for i, j, k in sigma:
        if flags[i, j, k] == FLAG_GAS:
            sigma[i, j, k] = sigma_gas
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
    flags: ti.template(),
    nz_solid: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
):
    """Gaussian current source on metal where phi is solved.

    Gas and the grounded plate are phi = 0, so source there would be dropped and
    elec_normalize_source could not deliver current_A.
    """
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma_cells * sigma_cells + eps)
    cell_vol = dx * dx * dx
    k_max = ti.min(nz, ti.cast(arc_k + sigma_cells * 4.0, ti.i32) + 1)
    k_min = ti.max(0, ti.cast(arc_k - sigma_cells, ti.i32))
    for i, j, k in source:
        if k < k_min or k > k_max:
            continue
        if _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID) == 0:
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


_ELEC_FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@ti.func
def _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID):
    """1 where phi is solved; gas and the grounded plate are phi = 0."""
    f = flags[i, j, k]
    out = 1
    if f == FLAG_GAS or (k <= nz_solid and f == FLAG_SOLID):
        out = 0
    return out


@ti.func
def _elec_face_sigma(sigma, i, j, k, ii, jj, kk):
    """Series (harmonic-mean) conductivity of the face between two cells."""
    a = sigma[i, j, k]
    b = sigma[ii, jj, kk]
    return 2.0 * a * b / (a + b + 1e-30)


@ti.func
def _elec_apply(x, sigma, flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID, nx, ny, nz):
    """(A x)_c = sum of sigma_f (x_c - x_n) over in-domain faces; x = 0 on Dirichlet cells.

    Domain faces are insulated. Returns (A x)_c and the diagonal sum of sigma_f.
    """
    xc = x[i, j, k]
    ax = 0.0
    diag = 0.0
    for f in ti.static(range(6)):
        ii = i + _ELEC_FACES[f][0]
        jj = j + _ELEC_FACES[f][1]
        kk = k + _ELEC_FACES[f][2]
        if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
            s_f = _elec_face_sigma(sigma, i, j, k, ii, jj, kk)
            xn = 0.0
            if _elec_unknown(flags, ii, jj, kk, nz_solid, FLAG_GAS, FLAG_SOLID) == 1:
                xn = x[ii, jj, kk]
            ax += s_f * (xc - xn)
            diag += s_f
    return ti.Vector([ax, diag])


@ti.kernel
def elec_pcg_init(
    phi: ti.template(),
    r: ti.template(),
    p: ti.template(),
    diag: ti.template(),
    sigma: ti.template(),
    source: ti.template(),
    flags: ti.template(),
    scal: ti.template(),
    box: ti.template(),
    dx: ti.f32,
    nz_solid: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """r = b - A phi, p = r / diag. scal: [0] r.z, [3] r.r, [4] b.b (b = source dx^2).

    box gets the inclusive index bounds of the unknown cells
    (i_min, i_max, j_min, j_max, k_min, k_max); i_max < 0 when there are none.
    """
    scal[0] = 0.0
    scal[3] = 0.0
    scal[4] = 0.0
    box[0] = nx
    box[1] = -1
    box[2] = ny
    box[3] = -1
    box[4] = nz
    box[5] = -1
    for i, j, k in phi:
        if _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID) == 0:
            phi[i, j, k] = 0.0
            r[i, j, k] = 0.0
            p[i, j, k] = 0.0
            diag[i, j, k] = 1.0
            continue
        ad = _elec_apply(phi, sigma, flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID, nx, ny, nz)
        d = ti.max(ad[1], 1e-30)
        b = source[i, j, k] * dx * dx
        rv = b - ad[0]
        z = rv / d
        diag[i, j, k] = d
        r[i, j, k] = rv
        p[i, j, k] = z
        ti.atomic_min(box[0], i)
        ti.atomic_max(box[1], i)
        ti.atomic_min(box[2], j)
        ti.atomic_max(box[3], j)
        ti.atomic_min(box[4], k)
        ti.atomic_max(box[5], k)
        scal[0] += ti.cast(rv * z, ti.f64)
        scal[3] += ti.cast(rv * rv, ti.f64)
        scal[4] += ti.cast(b * b, ti.f64)


@ti.kernel
def elec_pcg_iterate(
    phi: ti.template(),
    r: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
    diag: ti.template(),
    sigma: ti.template(),
    flags: ti.template(),
    scal: ti.template(),
    nz_solid: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_SOLID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    i0: ti.i32,
    i1: ti.i32,
    j0: ti.i32,
    j1: ti.i32,
    k0: ti.i32,
    k1: ti.i32,
    n_iter: ti.template(),
):
    """n_iter Jacobi-preconditioned CG iterations in one launch.

    Sweeps only the unknown-cell box [i0, i1) x [j0, j1) x [k0, k1); scal uses
    the elec_pcg_init layout plus [1] new r.z and [2] p.Ap. Top-level loops run
    in order with a device-wide sync between them, so each loop sees the
    previous loop's reductions.
    """
    for _ in ti.static(range(n_iter)):
        scal[2] = 0.0
        for i, j, k in ti.ndrange((i0, i1), (j0, j1), (k0, k1)):
            v = 0.0
            if _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID) == 1:
                v = _elec_apply(p, sigma, flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID, nx, ny, nz)[0]
                scal[2] += ti.cast(p[i, j, k] * v, ti.f64)
            Ap[i, j, k] = v
        scal[1] = 0.0
        scal[3] = 0.0
        for i, j, k in ti.ndrange((i0, i1), (j0, j1), (k0, k1)):
            if _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID) == 1:
                alpha = 0.0
                if scal[2] > 0.0:
                    alpha = ti.cast(scal[0] / scal[2], ti.f32)
                phi[i, j, k] += alpha * p[i, j, k]
                rv = r[i, j, k] - alpha * Ap[i, j, k]
                r[i, j, k] = rv
                scal[1] += ti.cast(rv * rv / diag[i, j, k], ti.f64)
                scal[3] += ti.cast(rv * rv, ti.f64)
        for i, j, k in ti.ndrange((i0, i1), (j0, j1), (k0, k1)):
            if _elec_unknown(flags, i, j, k, nz_solid, FLAG_GAS, FLAG_SOLID) == 1:
                beta = 0.0
                if scal[0] > 0.0:
                    beta = ti.cast(scal[1] / scal[0], ti.f32)
                p[i, j, k] = r[i, j, k] / diag[i, j, k] + beta * p[i, j, k]
        scal[0] = scal[1]


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
    """Cell J = mean of the two face currents per axis, sigma_f (phi_c - phi_n) / dx.

    A cell-sigma central difference straddling a conductivity jump (metal next to
    gas or the grounded plate) overstates J by up to sigma_cell / sigma_face.
    Insulated domain faces carry no current.
    """
    for i, j, k in Jx:
        if flags[i, j, k] == FLAG_GAS:
            Jx[i, j, k] = 0.0
            Jy[i, j, k] = 0.0
            Jz[i, j, k] = 0.0
            continue
        pc = phi[i, j, k]
        J = ti.Vector([0.0, 0.0, 0.0])
        for f in ti.static(range(6)):
            ii = i + _ELEC_FACES[f][0]
            jj = j + _ELEC_FACES[f][1]
            kk = k + _ELEC_FACES[f][2]
            if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                outward = _elec_face_sigma(sigma, i, j, k, ii, jj, kk) * (pc - phi[ii, jj, kk]) / dx
                sgn = ti.static(_ELEC_FACES[f][0] + _ELEC_FACES[f][1] + _ELEC_FACES[f][2])
                axis = ti.static(0 if _ELEC_FACES[f][0] != 0 else (1 if _ELEC_FACES[f][1] != 0 else 2))
                J[axis] += 0.5 * sgn * outward
        Jx[i, j, k] = J[0]
        Jy[i, j, k] = J[1]
        Jz[i, j, k] = J[2]


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
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
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
        rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
        scale = fl_w * dt * dt / (rho_ref * dx)
        Fx[i, j, k] += fx_phys * scale
        Fy[i, j, k] += fy_phys * scale
        Fz[i, j, k] += fz_phys * scale


