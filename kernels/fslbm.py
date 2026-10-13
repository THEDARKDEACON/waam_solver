"""
Koerner / Thürey free-surface LBM (Phase 3).

Interface cells reconstruct missing incoming populations from gas using a
pressure boundary condition (surface tension + arc/recoil pressure).
Mass of interface cells is tracked via fill fraction φ ∈ (0, 1).

This module is activated when ``twin.use_free_surface_lbm`` is True;
legacy CSF body-force path remains the default until benchmarks pass.
"""

from __future__ import annotations

from waam_twin.compiler import ti
from ..lattice import EX, EY, EZ, W, OPP


@ti.func
def _height_curvature_2d(phi, i, j, k, ny, nz):
    """
    Height-function curvature in the yz plane (dominant for a bead crown).

    κ ≈ −h_yy / (1 + h_y²)^{3/2} with h(y) = Σ_k φ(i,j,k)·dx (cell units).
    """
    # Column heights at j-1, j, j+1
    h_m = 0.0
    h_0 = 0.0
    h_p = 0.0
    for kk in range(nz):
        h_0 += phi[i, j, kk]
        if j > 0:
            h_m += phi[i, j - 1, kk]
        if j + 1 < ny:
            h_p += phi[i, j + 1, kk]
    if j == 0:
        h_m = h_0
    if j + 1 >= ny:
        h_p = h_0
    hy = 0.5 * (h_p - h_m)
    hyy = h_p - 2.0 * h_0 + h_m
    denom = (1.0 + hy * hy) ** 1.5 + 1e-9
    return -hyy / denom


@ti.kernel
def reconstruct_interface_populations(
    f: ti.template(),
    rho: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    phi: ti.template(),
    flags: ti.template(),
    P_extra: ti.template(),  # additional surface pressure [lattice]
    gamma_lu: ti.f32,
    FLAG_IFACE: ti.i32,
    FLAG_GAS: ti.i32,
    FLAG_FLUID: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    After streaming: for IFACE cells, replace populations that came from GAS
    with equilibrium at ρ_b = ρ_liquid + 3·(γ·κ + P_extra) (Koerner BC).
    """
    cs2 = 1.0 / 3.0
    for i, j, k in flags:
        if flags[i, j, k] != FLAG_IFACE:
            continue
        kappa = _height_curvature_2d(phi, i, j, k, ny, nz)
        P_cap = gamma_lu * kappa + P_extra[i, j, k]
        rho_b = rho[i, j, k] + 3.0 * P_cap  # p = cs² ρ → Δρ = 3 Δp
        ux_b = ux[i, j, k]
        uy_b = uy[i, j, k]
        uz_b = uz[i, j, k]
        u2 = ux_b * ux_b + uy_b * uy_b + uz_b * uz_b
        for q in ti.static(range(19)):
            ni = i - EX[q]
            nj = j - EY[q]
            nk = k - EZ[q]
            from_gas = False
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                from_gas = True
            elif flags[ni, nj, nk] == FLAG_GAS:
                from_gas = True
            if from_gas:
                eu = EX[q] * ux_b + EY[q] * uy_b + EZ[q] * uz_b
                feq = W[q] * rho_b * (
                    1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
                )
                # Koerner: f_q = f_eq(ρ_b, u) + f_eq(ρ_b, u) − f_opp_incoming
                # Simplified: set missing direction to equilibrium at ρ_b, u
                f[q, i, j, k] = feq


@ti.kernel
def update_fill_fraction(
    phi: ti.template(),
    rho: ti.template(),
    flags: ti.template(),
    mass: ti.template(),
    FLAG_IFACE: ti.i32,
    FLAG_FLUID: ti.i32,
    FLAG_GAS: ti.i32,
    rho0: ti.f32,
):
    """Advance interface fill fraction from mass / ρ0; convert emptied/filled cells."""
    for i, j, k in flags:
        if flags[i, j, k] != FLAG_IFACE:
            continue
        m = mass[i, j, k]
        fill = m / (rho0 + 1e-12)
        fill = ti.max(0.0, ti.min(1.0, fill))
        phi[i, j, k] = fill
        if fill >= 0.99:
            flags[i, j, k] = FLAG_FLUID
            phi[i, j, k] = 1.0
            mass[i, j, k] = rho0
        elif fill <= 0.01:
            flags[i, j, k] = FLAG_GAS
            phi[i, j, k] = 0.0
            mass[i, j, k] = 0.0
            rho[i, j, k] = 0.0


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
