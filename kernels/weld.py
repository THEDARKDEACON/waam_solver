"""Taichi kernels — recoil, gas shear, droplet, arc pressure."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP
from ._common import (
    _cell_rho,
)

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
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Vapor recoil pressure on the free surface (downward, T-dependent)."""
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)

    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue
        dphi_z = phi[i, j, ti.min(k + 1, Fz.shape[2] - 1)] - phi[i, j, ti.max(k - 1, 0)]
        if ti.abs(dphi_z) < 0.05:
            continue
        rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
        F_peak_lu = (recoil_pa / (rho_ref * dx)) * dt * dt / dx
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
    T_onset_K: ti.f32,
    L_vapor_J_kg: ti.f32,
    R_spec_J_kgK: ti.f32,
    C_acc: ti.f32,
    dt: ti.f32,
    dx: ti.f32,
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """
    Vapor recoil via Clausius–Clapeyron: p = C_acc · P_sat(T) · ramp².

    C_acc ≈ 0.54 (Anisimov / Knight accommodation). Soft quadratic onset from
    T_onset → T_boil (same schedule as evaporative cooling) so conduction-mode
    pools can develop a partial recoil force without needing T > T_boil.
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
        if Tc <= T_onset_K:
            continue
        ramp = 1.0
        if Tc < T_boil_K:
            ramp = (Tc - T_onset_K) / (T_boil_K - T_onset_K + eps)
            ramp = ti.max(0.0, ti.min(1.0, ramp))
            ramp = ramp * ramp
        exponent = (L_vapor_J_kg / (R_spec_J_kgK + eps)) * (1.0 / (T_boil_K + eps) - 1.0 / (Tc + eps))
        exponent = ti.max(ti.min(exponent, 12.0), -8.0)
        P_vap = C_acc * P_ref_Pa * ti.math.exp(exponent) * ramp
        rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
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
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)
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
        rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
        F_scale = _wf_pressure_to_Fz_lu(1.0, dt, dx, rho_ref)
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
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    eps = 1e-6
    inv2s2 = 1.0 / (2.0 * drop_radius * drop_radius + eps)
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
        rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
        F_peak_lu = _wf_pressure_to_Fz_lu(impact_pa, dt, dx, rho_ref)
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
    alloy_id: ti.template(),
    rho_wire: ti.f32,
    rho_plate: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS:   ti.i32,
):
    """
    Apply a downward Gaussian force representing arc pressure
    and droplet impact to the free surface.

    Lin–Eagar is an axisymmetric surface load: the Gaussian uses in-plane
    r² = di² + dj² (same as Clausius–Clapeyron recoil). Interface cells
    are selected by |∇φ_z| > 0.05.

    The force is mapped from physical pressure [Pa] to LBM body force [lu/ts²].
    F_phys = P_phys / (rho * dx)  [m/s²]
    F_lu   = F_phys * dt² / dx    [lu/ts²]
    """
    eps = 1e-6
    # eps guards against a zero-sigma call producing inf in inv2s2
    inv2s2 = 1.0 / (2.0 * sigma * sigma + eps)

    for i, j, k in Fz:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        # Apply pressure to free-surface interface cells.
        # Threshold 0.05 (down from 0.1) catches the full 2-cell-thick interface layer.
        # Lin–Eagar is an axisymmetric *surface* load: r² is in-plane only
        # (same convention as Clausius–Clapeyron recoil). |∇φ_z| already
        # restricts the dump to the gas-metal interface.
        dphi_z = phi[i, j, ti.min(k+1, Fz.shape[2]-1)] - phi[i, j, ti.max(k-1, 0)]
        if ti.abs(dphi_z) > 0.05:
            rho_ref = _cell_rho(alloy_id, i, j, k, rho_wire, rho_plate)
            F_peak_phys = pressure_pa / (rho_ref * dx)
            F_peak_lu   = F_peak_phys * dt**2 / dx
            di = ti.f32(i) - arc_i
            dj = ti.f32(j) - arc_j
            r2 = di * di + dj * dj
            force = -F_peak_lu * ti.math.exp(-r2 * inv2s2)
            Fz[i, j, k] += force


