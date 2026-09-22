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


