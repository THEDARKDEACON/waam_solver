"""
collision.py — Two-relaxation-time (TRT) D3Q19 collision
=========================================================
Replaces the previous Hermite-rebuild "central-moment MRT", which was
mass-unstable at τ→0.5005 (ρ blew to 1e9+ once liquid formed).

WHAT THIS IS (be precise in publications):
  D3Q19 TRT / two-rate collision (Ginzburg): even (symmetric) moments
  relax at ω_e = 1/τ_s (shear viscosity); odd (antisymmetric) moments
  relax at ω_o from the magic parameter Λ = (τ_s−½)(τ_o−½) = ¼.
  Higher-order behaviour is that of TRT, not a full multi-rate MRT or
  Geier cumulant operator. With ω_o = ω_e it reduces exactly to SRT.

Mass and momentum are conserved by construction (no f₀ dump).
Guo forcing + semi-implicit Guo–Darcy for body forces / mushy drag.
"""

from waam_twin.compiler import ti
from .lattice import EX, EY, EZ, W, OPP, Q


@ti.func
def _trt_omega_odd(omega_e: ti.f32) -> ti.f32:
    """Odd-rate from magic Λ=1/4 given even-rate ω_e = 1/τ_s."""
    tau_s = 1.0 / omega_e
    if tau_s < 0.5001:
        tau_s = 0.5001
    tau_o = 0.5 + 0.25 / (tau_s - 0.5)
    return 1.0 / tau_o


@ti.func
def _smagorinsky_tau(
    f_src: ti.template(),
    i: ti.i32, j: ti.i32, k: ti.i32,
    r: ti.f32,
    ux_f: ti.f32, uy_f: ti.f32, uz_f: ti.f32,
    fx: ti.f32, fy: ti.f32, fz: ti.f32,
    tau0: ti.f32,
    cs_smag: ti.f32,
) -> ti.f32:
    """Smagorinsky-raised relaxation time from the local non-equilibrium stress.

    tau = (tau0 + sqrt(tau0^2 + 18 sqrt(2) Cs^2 |Pi_neq| / rho)) / 2 (Hou et
    al. 1996, lattice units). The added eddy viscosity scales with the cell
    size squared, so it fades as the mesh is refined and Cs is not mesh-tuned.
    """
    u2 = ux_f * ux_f + uy_f * uy_f + uz_f * uz_f
    pxx = 0.0
    pyy = 0.0
    pzz = 0.0
    pxy = 0.0
    pxz = 0.0
    pyz = 0.0
    for q in ti.static(range(Q)):
        ex_q = ti.cast(EX[q], ti.f32)
        ey_q = ti.cast(EY[q], ti.f32)
        ez_q = ti.cast(EZ[q], ti.f32)
        eu = ex_q * ux_f + ey_q * uy_f + ez_q * uz_f
        fneq = f_src[q, i, j, k] - ti.cast(W[q], ti.f32) * r * (
            1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2
        )
        pxx += ex_q * ex_q * fneq
        pyy += ey_q * ey_q * fneq
        pzz += ez_q * ez_q * fneq
        pxy += ex_q * ey_q * fneq
        pxz += ex_q * ez_q * fneq
        pyz += ey_q * ez_q * fneq
    # Guo forcing shifts the moment by half a step of F u + u F.
    pxx += fx * ux_f
    pyy += fy * uy_f
    pzz += fz * uz_f
    pxy += 0.5 * (fx * uy_f + fy * ux_f)
    pxz += 0.5 * (fx * uz_f + fz * ux_f)
    pyz += 0.5 * (fy * uz_f + fz * uy_f)
    pmag = ti.sqrt(
        pxx * pxx + pyy * pyy + pzz * pzz
        + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz)
    )
    return 0.5 * (tau0 + ti.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * cs_smag * cs_smag * pmag / (r + 1e-6)))


@ti.func
def _trt_collide_cell(
    f_src: ti.template(),
    f_dst: ti.template(),
    i: ti.i32, j: ti.i32, k: ti.i32,
    r: ti.f32,
    ux_f: ti.f32, uy_f: ti.f32, uz_f: ti.f32,
    fx: ti.f32, fy: ti.f32, fz: ti.f32,
    omega_e: ti.f32, omega_o: ti.f32,
):
    u2 = ux_f * ux_f + uy_f * uy_f + uz_f * uz_f
    guo_pre = 1.0 - 0.5 * omega_e

    for q in ti.static(range(Q)):
        ex_q = ti.cast(EX[q], ti.f32)
        ey_q = ti.cast(EY[q], ti.f32)
        ez_q = ti.cast(EZ[q], ti.f32)
        w_q = ti.cast(W[q], ti.f32)
        eu = ex_q * ux_f + ey_q * uy_f + ez_q * uz_f
        feq_q = w_q * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
        F_dot_e = ex_q * fx + ey_q * fy + ez_q * fz
        S_q = w_q * guo_pre * (
            3.0 * F_dot_e
            + 9.0 * (eu * F_dot_e)
            - 3.0 * (ux_f * fx + uy_f * fy + uz_f * fz)
        )
        opp = ti.static(OPP[q])
        if ti.static(q == 0):
            f_dst[0, i, j, k] = (
                f_src[0, i, j, k]
                - omega_e * (f_src[0, i, j, k] - feq_q)
                + S_q
            )
        else:
            ex_o = ti.cast(EX[opp], ti.f32)
            ey_o = ti.cast(EY[opp], ti.f32)
            ez_o = ti.cast(EZ[opp], ti.f32)
            w_o = ti.cast(W[opp], ti.f32)
            eu_o = ex_o * ux_f + ey_o * uy_f + ez_o * uz_f
            feq_o = w_o * r * (1.0 + 3.0 * eu_o + 4.5 * eu_o * eu_o - 1.5 * u2)
            f_s = 0.5 * (f_src[q, i, j, k] + f_src[opp, i, j, k])
            f_a = 0.5 * (f_src[q, i, j, k] - f_src[opp, i, j, k])
            feq_s = 0.5 * (feq_q + feq_o)
            feq_a = 0.5 * (feq_q - feq_o)
            f_dst[q, i, j, k] = (
                f_src[q, i, j, k]
                - omega_e * (f_s - feq_s)
                - omega_o * (f_a - feq_a)
                + S_q
            )


@ti.kernel
def collide_mrt(
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
    EX_f: ti.template(), EY_f: ti.template(), EZ_f: ti.template(),
    W_f:  ti.template(),
    OPP_f: ti.template(),
    omega_s:     ti.f32,
    omega_b:     ti.f32,
    C_darcy:     ti.f32,
    FLAG_SOLID:  ti.i32,
    FLAG_GAS:    ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
):
    """
    Constant-τ TRT collision (API name kept as collide_mrt).

    omega_s → even (shear) rate. If omega_b ≈ omega_s, odd rate uses
    magic Λ=1/4; otherwise omega_b is honoured as the odd rate.
    Field velocity-set args kept for call-site compatibility.
    """
    eps = 1.0e-6
    omega_e = omega_s
    omega_o = ti.select(
        ti.abs(omega_b - omega_s) < 1.0e-3,
        _trt_omega_odd(omega_e),
        omega_b,
    )

    for i, j, k in rho:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        fl = f_l[i, j, k]
        fx = Fx[i, j, k]
        fy = Fy[i, j, k]
        fz = Fz[i, j, k]

        r = 0.0
        jx = 0.0
        jy = 0.0
        jz = 0.0
        for q in ti.static(range(Q)):
            fq = f_src[q, i, j, k]
            r += fq
            jx += fq * ti.cast(EX[q], ti.f32)
            jy += fq * ti.cast(EY[q], ti.f32)
            jz += fq * ti.cast(EZ[q], ti.f32)

        inv_r = 1.0 / (r + eps)
        ux_raw = (jx + 0.5 * fx) * inv_r
        uy_raw = (jy + 0.5 * fy) * inv_r
        uz_raw = (jz + 0.5 * fz) * inv_r

        K_darcy = C_darcy * ((1.0 - fl) ** 2) / (fl ** 3 + eps)
        ck_d = 1.0 + 0.5 * K_darcy
        ux_f = ux_raw / ck_d
        uy_f = uy_raw / ck_d
        uz_f = uz_raw / ck_d
        fx = fx - K_darcy * ux_f
        fy = fy - K_darcy * uy_f
        fz = fz - K_darcy * uz_f

        _trt_collide_cell(
            f_src, f_dst, i, j, k,
            r, ux_f, uy_f, uz_f, fx, fy, fz, omega_e, omega_o,
        )
        rho[i, j, k] = r
        ux[i, j, k] = ux_f
        uy[i, j, k] = uy_f
        uz[i, j, k] = uz_f


@ti.kernel
def collide_mrt_variable_tau(
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
    tau_field: ti.template(),
    flags: ti.template(),
    EX_f: ti.template(), EY_f: ti.template(), EZ_f: ti.template(),
    W_f:  ti.template(),
    OPP_f: ti.template(),
    omega_b:     ti.f32,
    C_darcy:     ti.f32,
    FLAG_SOLID:  ti.i32,
    FLAG_GAS:    ti.i32,
    nx: ti.i32, ny: ti.i32, nz: ti.i32,
    cs_smag: ti.f32,
):
    """
    Per-cell τ_s TRT collision (API name kept as collide_mrt_variable_tau).

    omega_s = 1/τ_eff with τ_eff = τ_field raised by the Smagorinsky eddy
    viscosity when cs_smag > 0. Odd rate from magic Λ=1/4 when
    omega_b ≈ 1/τ_field (production default); otherwise omega_b is used as ω_o.
    """
    eps = 1.0e-6

    for i, j, k in rho:
        if flags[i, j, k] == FLAG_SOLID or flags[i, j, k] == FLAG_GAS:
            continue

        fl = f_l[i, j, k]
        tau_loc = tau_field[i, j, k]
        if tau_loc < 0.5001:
            tau_loc = 0.5001
        magic_odd = ti.abs(omega_b - 1.0 / tau_loc) < 1.0e-3

        fx = Fx[i, j, k]
        fy = Fy[i, j, k]
        fz = Fz[i, j, k]

        r = 0.0
        jx = 0.0
        jy = 0.0
        jz = 0.0
        for q in ti.static(range(Q)):
            fq = f_src[q, i, j, k]
            r += fq
            jx += fq * ti.cast(EX[q], ti.f32)
            jy += fq * ti.cast(EY[q], ti.f32)
            jz += fq * ti.cast(EZ[q], ti.f32)

        inv_r = 1.0 / (r + eps)
        ux_raw = (jx + 0.5 * fx) * inv_r
        uy_raw = (jy + 0.5 * fy) * inv_r
        uz_raw = (jz + 0.5 * fz) * inv_r

        K_darcy = C_darcy * ((1.0 - fl) ** 2) / (fl ** 3 + eps)
        ck_d = 1.0 + 0.5 * K_darcy
        ux_f = ux_raw / ck_d
        uy_f = uy_raw / ck_d
        uz_f = uz_raw / ck_d
        fx = fx - K_darcy * ux_f
        fy = fy - K_darcy * uy_f
        fz = fz - K_darcy * uz_f

        if cs_smag > 0.0:
            tau_loc = _smagorinsky_tau(
                f_src, i, j, k, r, ux_f, uy_f, uz_f, fx, fy, fz, tau_loc, cs_smag,
            )
        omega_e = 1.0 / tau_loc
        omega_o = ti.select(magic_odd, _trt_omega_odd(omega_e), omega_b)

        _trt_collide_cell(
            f_src, f_dst, i, j, k,
            r, ux_f, uy_f, uz_f, fx, fy, fz, omega_e, omega_o,
        )
        rho[i, j, k] = r
        ux[i, j, k] = ux_f
        uy[i, j, k] = uy_f
        uz[i, j, k] = uz_f
