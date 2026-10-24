"""
test_fslbm.py — free-surface LBM (Körner/Thürey) mass conservation and motion.

Dam break: a liquid column in a closed box collapses under gravity. Metal
mass Σ φ·ρ must be conserved and the front must advance. Flat pool: a level
surface with surface tension and no gravity must stay quiescent. Laplace: a
droplet's core pressure must exceed the gas pressure by 2γ/R.
"""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin, kernels
from waam_twin.solvers.coupled_step import _free_surface_stream


def _setup(nx: int, ny: int, nz: int, liquid) -> WAAMTwin:
    twin = WAAMTwin(nx=nx, ny=ny, nz=nz, dx=3e-4, max_tracers=10)
    twin.reset(T_ambient=float(twin.mat.T_liquidus) + 200.0, test_fluid_domain=True)
    g = twin.grid
    g.ensure_fslbm_fields()
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    wet = liquid(i, j, k)
    flags = np.where(wet, g.FLAG_FLUID, g.FLAG_GAS).astype(np.int32)
    g.flags.from_numpy(flags)
    g.phi.from_numpy(wet.astype(np.float32))
    g.f_l.from_numpy(np.ones((nx, ny, nz), np.float32))
    return twin


def _snap(g) -> dict:
    names = (
        "f_src", "f_dst", "flags", "phi", "f_l", "H", "T", "rho",
        "ux", "uy", "uz", "fs_mass", "fs_energy", "fs_mark",
    )
    out = {name: getattr(g, name).to_numpy().copy() for name in names}
    out["acc"] = g.energy_acc_J.to_numpy().copy()
    out["lost"] = float(g.fs_mass_lost[None])
    return out


def _restore(g, snap: dict) -> None:
    for name, values in snap.items():
        if name == "acc":
            g.energy_acc_J.from_numpy(values)
        elif name == "lost":
            g.fs_mass_lost[None] = values
        else:
            getattr(g, name).from_numpy(values)


def _metal_energy(g) -> float:
    flags = g.flags.to_numpy()
    w = np.clip(g.phi.to_numpy().astype("f8"), 0.0, 1.0)
    return float((w * g.H.to_numpy().astype("f8"))[flags != g.FLAG_GAS].sum())


def _metal_mass(g) -> float:
    flags = g.flags.to_numpy()
    m = g.phi.to_numpy().astype("f8") * g.rho.to_numpy().astype("f8")
    return float(m[flags != g.FLAG_GAS].sum())


def _run(twin: WAAMTwin, n: int, g_lu: float, gamma_lu: float, tau: float) -> None:
    g = twin.grid
    twin.enable_csf_tension = gamma_lu > 0.0
    twin.gamma_lu = gamma_lu
    for _ in range(n):
        kernels.clear_forces(g.Fx, g.Fy, g.Fz)
        if g_lu > 0.0:
            fz = g.Fz.to_numpy()
            fz[g.flags.to_numpy() != g.FLAG_GAS] = -g_lu
            g.Fz.from_numpy(fz)
        kernels.collide_srt(
            g.f_src, g.f_dst, g.rho, g.ux, g.uy, g.uz, g.Fx, g.Fy, g.Fz,
            g.f_l, g.flags, tau, 1.0 / tau, 1.0, 0.0,
            g.FLAG_SOLID, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
        _free_surface_stream(twin, g)


def dam_break(n: int = 600, tol: float = 2e-3) -> None:
    nx, ny, nz = 32, 8, 20
    twin = _setup(nx, ny, nz, lambda i, j, k: (i < 10) & (k < 14))
    g = twin.grid
    H = g.H.to_numpy()
    H *= (1.0 + 0.05 * np.arange(nx, dtype=np.float32))[:, None, None]
    g.H.from_numpy(H)
    m0 = _metal_mass(g)
    E0 = _metal_energy(g)
    acc0 = float(g.energy_acc_J.to_numpy()[3])
    snap = _snap(g)
    _run(twin, n, g_lu=2e-4, gamma_lu=0.0, tau=0.6)
    flags_a = g.flags.to_numpy().copy()
    phi_a = g.phi.to_numpy().copy()
    _restore(g, snap)
    _run(twin, n, g_lu=2e-4, gamma_lu=0.0, tau=0.6)
    if not np.array_equal(flags_a, g.flags.to_numpy()) or not np.array_equal(phi_a, g.phi.to_numpy()):
        raise AssertionError("dam break flags/phi differed between identical runs")
    m1 = _metal_mass(g)
    booked = (float(g.energy_acc_J.to_numpy()[3]) - acc0) / g.dx ** 3
    e_rel = abs(_metal_energy(g) - E0 - booked) / E0
    flags = g.flags.to_numpy()
    wet_i = np.nonzero((flags != g.FLAG_GAS).any(axis=(1, 2)))[0]
    front = int(wet_i.max()) if wet_i.size else -1
    lost = float(g.fs_mass_lost[None])
    finite = bool(np.isfinite(g.f_src.to_numpy()).all())
    rel = abs(m1 - m0) / m0
    print(
        f"[fslbm] dam break  mass {m0:.2f} -> {m1:.2f}  rel={rel:.2e}  "
        f"unplaced={lost:.2e}  front i={front} (started 9)  finite={finite}  "
        f"energy closure={e_rel:.1e} (booked {booked / E0:.1e} of E)"
    )
    if not finite:
        raise AssertionError("dam break went non-finite")
    if rel > tol:
        raise AssertionError(f"metal mass drift {rel:.2e} > {tol:.0e}")
    if front < 14:
        raise AssertionError(f"dam front did not advance (i={front})")
    if e_rel > 1e-4:
        raise AssertionError(f"interface energy not conserved ({e_rel:.1e})")


def flat_pool(n: int = 400, u_tol: float = 1e-4) -> None:
    nx, ny, nz = 16, 16, 16
    twin = _setup(nx, ny, nz, lambda i, j, k: k < 8)
    g = twin.grid
    m0 = _metal_mass(g)
    _run(twin, n, g_lu=0.0, gamma_lu=0.01, tau=0.6)
    flags = g.flags.to_numpy()
    metal = flags != g.FLAG_GAS
    u = np.sqrt(g.ux.to_numpy() ** 2 + g.uy.to_numpy() ** 2 + g.uz.to_numpy() ** 2)
    u_max = float(u[metal].max())
    rel = abs(_metal_mass(g) - m0) / m0
    print(f"[fslbm] flat pool  u_max={u_max:.2e}  mass rel={rel:.2e}")
    if u_max > u_tol:
        raise AssertionError(f"spurious currents {u_max:.2e} > {u_tol:.0e}")


def laplace_droplet(n: int = 1000, tol: float = 0.10) -> None:
    N, R, gamma = 28, 7.0, 0.01
    c = (N - 1) / 2
    twin = _setup(N, N, N, lambda i, j, k: (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2 < R * R)
    g = twin.grid
    _run(twin, n, g_lu=0.0, gamma_lu=gamma, tau=0.8)
    flags = g.flags.to_numpy()
    vol = float((g.phi.to_numpy() * (flags != g.FLAG_GAS)).sum())
    R_eff = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    dp = (float(g.rho.to_numpy()[flags == g.FLAG_FLUID].mean()) - 1.0) / 3.0
    ratio = dp / (2.0 * gamma / R_eff)
    print(f"[fslbm] laplace  R={R_eff:.2f}  dp/(2γ/R)={ratio:.3f}")
    if abs(ratio - 1.0) > tol:
        raise AssertionError(f"Laplace pressure off by {ratio - 1.0:+.1%}")


def droplet_repeat() -> None:
    """One droplet spent in index order fills the same cells twice."""
    nx = ny = nz = 8
    twin = _setup(nx, ny, nz, lambda i, j, k: k <= 2)
    g = twin.grid
    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()
    flags[:, :, 3] = g.FLAG_IFACE
    phi[:, :, 3] = 0.4
    g.flags.from_numpy(flags)
    g.phi.from_numpy(phi)
    cell = float(g.dx ** 3)

    def once() -> None:
        g.deposit_vol_buf[None] = 0.0
        g.deposit_real_buf[None] = 0.0
        kernels.fs_deposit_into_interface(
            g.phi, g.flags, g.H,
            3.5, 3.5, 3.0, 2.5, 1.5, 6.0 * cell, 1.0e6,
            g.deposit_vol_buf, g.deposit_real_buf, cell,
            g.FLAG_IFACE, nx, ny, nz,
        )
        kernels.feed_wire_surface(
            g.f_src, g.flags, g.f_l, g.phi, g.H, g.T, g.rho,
            3.5, 3.5, 3.0, 2.5, 1.5, 10.0 * cell,
            2500.0, float(twin.cp_rho), 2.0e8, 1.0,
            g.deposit_vol_buf, g.deposit_real_buf, cell,
            g.FLAG_GAS, g.FLAG_FLUID, g.FLAG_SOLID, nx, ny, nz,
        )

    snap = _snap(g)
    once()
    flags_a = g.flags.to_numpy().copy()
    phi_a = g.phi.to_numpy().copy()
    _restore(g, snap)
    once()
    if not np.array_equal(flags_a, g.flags.to_numpy()) or not np.array_equal(phi_a, g.phi.to_numpy()):
        raise AssertionError("droplet filled different cells on the second call")
    if int((g.flags.to_numpy() == g.FLAG_FLUID).sum()) <= int((snap["flags"] == g.FLAG_FLUID).sum()):
        raise AssertionError("droplet did not place any metal")
    print("[fslbm] droplet repeat  identical fill")


def run() -> None:
    init_taichi(backend="cpu")
    dam_break()
    droplet_repeat()
    flat_pool()
    laplace_droplet()


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
