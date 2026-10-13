"""
Phase 5 — conservative finite-volume thermal operators.

Advection uses a TVD (minmod) flux; diffusion is div(k ∇T).
"""

from __future__ import annotations

from waam_twin.compiler import ti


@ti.func
def _minmod(a, b):
    s = 0.0
    if a * b > 0.0:
        s = ti.min(ti.abs(a), ti.abs(b)) * (1.0 if a > 0.0 else -1.0)
    return s


@ti.func
def _face_flux_tvd(u_face, H_L, H_C, H_R):
    """Donor-cell with minmod limited high-order correction."""
    if u_face >= 0.0:
        slope = _minmod(H_C - H_L, H_R - H_C)
        H_up = H_C + 0.5 * slope
        return u_face * H_up
    slope = _minmod(H_R - H_C, H_C - H_L)
    H_up = H_C - 0.5 * slope
    return u_face * H_up


@ti.kernel
def advect_diffuse_conservative(
    H: ti.template(),
    H_src: ti.template(),
    T: ti.template(),
    ux: ti.template(),
    uy: ti.template(),
    uz: ti.template(),
    flags: ti.template(),
    k_field: ti.template(),  # conductivity [W/(m·K)] scaled to lattice
    cp_rho_field: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
    FLAG_SOLID: ti.i32,
    FLAG_GAS: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
):
    """
    Conservative update: ∂H/∂t + ∇·(u H) = ∇·(k ∇T).

    Diffusion uses face-averaged k; advection uses TVD fluxes. Writes H from H_src.
    """
    inv_dx = 1.0 / dx
    for i, j, k in H:
        if flags[i, j, k] == FLAG_GAS:
            continue
        # Face velocities (lattice units; dt=1 step)
        # x-faces
        flux = 0.0
        # −x
        if i > 0 and flags[i - 1, j, k] != FLAG_GAS:
            u_f = 0.5 * (ux[i - 1, j, k] + ux[i, j, k])
            HL = H_src[i - 1, j, k] if i > 1 else H_src[i - 1, j, k]
            HC = H_src[i - 1, j, k]
            HR = H_src[i, j, k]
            flux -= _face_flux_tvd(u_f, HL, HC, HR)
        # +x
        if i + 1 < nx and flags[i + 1, j, k] != FLAG_GAS:
            u_f = 0.5 * (ux[i, j, k] + ux[i + 1, j, k])
            HL = H_src[i, j, k]
            HC = H_src[i, j, k]
            HR = H_src[i + 1, j, k]
            flux += _face_flux_tvd(u_f, HL, HC, HR)
        # y
        if j > 0 and flags[i, j - 1, k] != FLAG_GAS:
            u_f = 0.5 * (uy[i, j - 1, k] + uy[i, j, k])
            flux -= _face_flux_tvd(
                u_f, H_src[i, j - 1, k], H_src[i, j - 1, k], H_src[i, j, k]
            )
        if j + 1 < ny and flags[i, j + 1, k] != FLAG_GAS:
            u_f = 0.5 * (uy[i, j, k] + uy[i, j + 1, k])
            flux += _face_flux_tvd(
                u_f, H_src[i, j, k], H_src[i, j, k], H_src[i, j + 1, k]
            )
        # z
        if k > 0 and flags[i, j, k - 1] != FLAG_GAS:
            u_f = 0.5 * (uz[i, j, k - 1] + uz[i, j, k])
            flux -= _face_flux_tvd(
                u_f, H_src[i, j, k - 1], H_src[i, j, k - 1], H_src[i, j, k]
            )
        if k + 1 < nz and flags[i, j, k + 1] != FLAG_GAS:
            u_f = 0.5 * (uz[i, j, k] + uz[i, j, k + 1])
            flux += _face_flux_tvd(
                u_f, H_src[i, j, k], H_src[i, j, k], H_src[i, j, k + 1]
            )

        # Diffusion: div(k ∇T)
        Tc = T[i, j, k]
        kc = k_field[i, j, k]
        diff = 0.0
        if i > 0 and flags[i - 1, j, k] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i - 1, j, k])
            diff += kf * (T[i - 1, j, k] - Tc) * inv_dx * inv_dx
        if i + 1 < nx and flags[i + 1, j, k] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i + 1, j, k])
            diff += kf * (T[i + 1, j, k] - Tc) * inv_dx * inv_dx
        if j > 0 and flags[i, j - 1, k] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i, j - 1, k])
            diff += kf * (T[i, j - 1, k] - Tc) * inv_dx * inv_dx
        if j + 1 < ny and flags[i, j + 1, k] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i, j + 1, k])
            diff += kf * (T[i, j + 1, k] - Tc) * inv_dx * inv_dx
        if k > 0 and flags[i, j, k - 1] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i, j, k - 1])
            diff += kf * (T[i, j, k - 1] - Tc) * inv_dx * inv_dx
        if k + 1 < nz and flags[i, j, k + 1] != FLAG_GAS:
            kf = 0.5 * (kc + k_field[i, j, k + 1])
            diff += kf * (T[i, j, k + 1] - Tc) * inv_dx * inv_dx

        # k is already in lattice enthalpy units (W/mK → via tables); dt is
        # one lattice step (=1) when called from coupled_step.
        H[i, j, k] = H_src[i, j, k] - dt * flux * inv_dx + dt * diff
