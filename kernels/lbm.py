"""Taichi kernels — LBM collide / stream / clamps."""

from waam_twin.compiler import ti

from ..lattice import EX, EY, EZ, W, OPP
from .vof import _brackbill_curvature_at

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


@ti.kernel
def clamp_body_force_magnitude(
    Fx: ti.template(),
    Fy: ti.template(),
    Fz: ti.template(),
    flags: ti.template(),
    F_max_lu: ti.f32,
    hit_buf: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Limit |F| so Guo forcing cannot drive Ma ≫ 0.1 in one step.

    ``hit_buf`` accumulates the number of fluid cells clamped this call.
    """
    eps = 1e-12
    hit_buf[None] = 0
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
            ti.atomic_add(hit_buf[None], 1)


@ti.kernel
def clamp_velocity_mach(
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    u_max_lu: ti.f32,
    hit_buf: ti.template(),
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
):
    """Cap macroscopic lattice velocity (keeps VOF / telemetry physical).

    ``hit_buf`` accumulates the number of fluid cells clamped this call.
    """
    eps = 1e-12
    hit_buf[None] = 0
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
            ti.atomic_add(hit_buf[None], 1)


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
            # Positive while cooling (metallurgical sign). Ignore absurd
            # leftovers (pre-filter spikes / NaN) above 1e5 K/s.
            cool = -dT_dt[i, j, k]
            if cool > 0.0 and cool < 1.0e5:
                ti.atomic_max(cool_buf[None], cool)
        if f_l[i, j, k] > 0.5:
            ti.atomic_add(n_buf[None], 1)
            ti.atomic_max(T_peak_buf[None], Tc)
            u2 = ux[i, j, k] * ux[i, j, k] + uy[i, j, k] * uy[i, j, k] + uz[i, j, k] * uz[i, j, k]
            ti.atomic_max(u2_buf[None], u2)
            ti.atomic_add(i_sum_buf[None], ti.f32(i))
            ti.atomic_min(i_min_buf[None], ti.f32(i))
            ti.atomic_max(i_max_buf[None], ti.f32(i))


@ti.kernel
def count_near_vapor_cap(
    T: ti.template(),
    flags: ti.template(),
    n_buf: ti.template(),
    T_cap: ti.f32,
    margin_K: ti.f32,
    FLAG_GAS: ti.i32,
):
    """Count metal cells with T ≥ T_cap − margin (enthalpy-ceiling saturation)."""
    n_buf[None] = 0
    thresh = T_cap - margin_K
    for i, j, k in T:
        if flags[i, j, k] == FLAG_GAS:
            continue
        if T[i, j, k] >= thresh:
            ti.atomic_add(n_buf[None], 1)


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
