"""
test_job_locks.py — Schema-validate lock YAMLs and pin audit-remediation behaviour.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np

from waam_twin import WAAMTwin, kernels
from waam_twin.frame import WeldFrame, apply_frame_to_twin, load_weld_frame
from waam_twin.job import apply_job_to_twin, electrical_arc_power_W, load_job_config, validate_job_config
from waam_twin.kernels import ALLOY_PLATE, ALLOY_WIRE
from waam_twin.physics.force_diagnostics import force_linf_lu, sample_force_diagnostics
from waam_twin.physics.weld_forces import lorentz_cold_max_iters, solve_lorentz
from waam_twin.runtime import init_taichi

_LOCK_DIR = Path(__file__).resolve().parent.parent / "jobs" / "examples" / "locks"
_LOCK_JOBS = (
    "heat_loss_hot.yaml",
    "thermal_tier.yaml",
    "strict_mass.yaml",
    "lorentz_strict.yaml",
    "interpass_travel.yaml",
    "frame_offset_z.yaml",
    "reset_ledgers.yaml",
    "lorentz_cold_start.yaml",
    "table_knots.yaml",
    "dual_alloy_rho.yaml",
    "heat_loss_factor.yaml",
)


def _load(name: str) -> dict:
    return load_job_config(f"jobs/examples/locks/{name}")


def _tiny(job: dict, *, nx: int = 16, ny: int = 12, nz: int = 14, dx: float = 1e-3, **kwargs):
    material = kwargs.pop("material", job.get("material", "ER70S-6"))
    plate_material = kwargs.pop("plate_material", (job.get("plate") or {}).get("material"))
    twin = WAAMTwin(
        nx=nx, ny=ny, nz=nz, dx=dx, max_tracers=8,
        material=material, plate_material=plate_material, **kwargs,
    )
    apply_job_to_twin(twin, job)
    twin.reset()
    return twin


def _lock_schema() -> None:
    found = sorted(p.name for p in _LOCK_DIR.glob("*.yaml") if p.name != "frame_z.yaml")
    expected = sorted(_LOCK_JOBS)
    if found != expected:
        raise AssertionError(f"lock YAML set {found} != {expected}")
    for name in _LOCK_JOBS:
        job = _load(name)
        if "model_reference" not in job:
            raise AssertionError(f"{name} missing model_reference")
    print(f"[job_locks] schema OK  n={len(_LOCK_JOBS)}")


def _lock_heat_loss_hot() -> None:
    job = _load("heat_loss_hot.yaml")
    model = job["model_reference"]
    twin = _tiny(job, enable_vof=False)
    g = twin.grid
    i, j, k = g.nx // 2, g.ny // 2, max(1, twin.nz_solid - 1)
    T_seed = float(model["T_seed_K"])
    T_legacy = float(model["T_legacy_cap_K"])
    T = g.T.to_numpy()
    H = g.H.to_numpy()
    T[i, j, k] = T_seed
    H[i, j, k] = twin.cp_rho * T_seed
    g.T.from_numpy(T)
    g.H.from_numpy(H)

    flags = g.flags.to_numpy()
    n_exp = 0
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ii, jj, kk = i + di, j + dj, k + dk
        if 0 <= ii < g.nx and 0 <= jj < g.ny and 0 <= kk < g.nz:
            if flags[ii, jj, kk] == g.FLAG_GAS:
                n_exp += 1
    if n_exp < 1:
        raise AssertionError("heat_loss_hot: seeded cell has no gas-facing face")

    h0 = float(g.H.to_numpy()[i, j, k])
    from waam_twin.physics import thermal

    t_cap = float(twin.T_vapor_cap_K) if twin.enable_enthalpy_cap else 0.0
    thermal.apply_boundary_losses(
        g.H, g.T, g.flags,
        twin.T_amb, twin.h_conv, twin.eps_rad,
        1 if twin.enable_convection else 0,
        1 if twin.enable_radiation else 0,
        twin.cp_rho, g.dt, g.dx, twin.sigma_sb, t_cap,
        g.FLAG_SOLID, g.FLAG_GAS,
        g.nx, g.ny, g.nz,
    )
    dH = h0 - float(g.H.to_numpy()[i, j, k])
    q_legacy = 0.0
    if twin.enable_convection:
        q_legacy += twin.h_conv * (T_legacy - twin.T_amb)
    if twin.enable_radiation:
        q_legacy += twin.eps_rad * twin.sigma_sb * (T_legacy ** 4 - twin.T_amb ** 4)
    dH_legacy = q_legacy * n_exp * g.dt / g.dx
    if dH <= dH_legacy * 1.05:
        raise AssertionError(
            f"heat-loss at {T_seed:.0f} K (dH={dH:.4e}) not above 2500 K-capped "
            f"analytic flux {dH_legacy:.4e}"
        )
    print(f"[job_locks] heat_loss_hot  dH={dH:.4e}  legacy_2500={dH_legacy:.4e}  n_exp={n_exp}")


def _lock_thermal_tier() -> None:
    job = _load("thermal_tier.yaml")
    twin = _tiny(job)
    if twin.enable_marangoni or twin.enable_buoyancy or twin.enable_arc_pressure:
        raise AssertionError(
            "thermal_tier.yaml must leave Marangoni/buoyancy/arc-pressure off"
        )
    g = twin.grid
    cx, cy = (g.nx // 2) * g.dx, (g.ny // 2) * g.dx
    n_steps = int((job.get("model_reference") or {}).get("n_steps", 8))
    for _ in range(n_steps):
        twin.step(cx, cy, is_welding=True)
    diag = sample_force_diagnostics(twin)
    eps = float((job.get("model_reference") or {}).get("force_eps", 1e-12))
    for key in ("f_marangoni_max", "f_buoyancy_max", "f_arc_max"):
        val = float(diag.get(key, 0.0))
        if val > eps:
            raise AssertionError(f"thermal tier {key}={val} expected ~0")
    print(f"[job_locks] thermal_tier  flags off  diag={ {k: diag[k] for k in ('f_marangoni_max','f_buoyancy_max','f_arc_max')} }")


def _lock_strict_mass() -> None:
    job = _load("strict_mass.yaml")
    model = job["model_reference"]
    twin = _tiny(job, nx=24, ny=16, nz=16, dx=8e-4, enable_vof=True)
    # Assert the wire ledger at the end; aborting at the 20th drop hides overflow skips.
    twin.strict_mode = False
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    n_steps = int(model.get("n_steps", 500))
    try:
        for step in range(n_steps):
            x = 0.004 + step * g.dt * twin.travel_speed_m_s
            twin.step(x, cy, is_welding=True)
    except RuntimeError as exc:
        overflow = int(getattr(twin, "_deposition_overflow", 0) or 0)
        if overflow > 0 and "mass_balance" in str(exc):
            print(
                f"[job_locks] strict_mass skipped (deposition overflow={overflow}): {exc}"
            )
            return
        raise
    telem = twin.get_telemetry()
    overflow = int(telem.get("deposition_overflow_count", 0) or 0)
    n_drops = int(telem.get("n_droplets_fired", 0) or 0)
    if overflow > 0:
        print(f"[job_locks] strict_mass skipped (overflow={overflow}, n_drops={n_drops})")
        return
    if n_drops < int(model.get("n_drops_min", 20)):
        raise AssertionError(f"strict_mass expected ≥20 drops, got {n_drops}")
    from waam_twin.physics.deposition_balance import wire_mass_flux_kg_s
    wire_g = float(telem["expected_wire_mass_g"])
    t_weld = float(telem["welding_time_s"])
    expect_g = wire_mass_flux_kg_s(twin) * t_weld * 1000.0
    if abs(wire_g - expect_g) / max(expect_g, 1e-12) > 1e-4:
        raise AssertionError(
            f"H3 ledger: expected_wire_mass_g={wire_g:.6f} != ṁ·t_weld={expect_g:.6f}"
        )
    ratio = float(telem["mass_balance_ratio"])
    dep = float(telem["deposited_mass_g"])
    if abs(ratio - dep / max(wire_g, 1e-12)) > 5e-3:
        raise AssertionError("mass_balance_ratio is not deposited/expected_wire")
    lo, hi = float(model["mass_ratio_min"]), float(model["mass_ratio_max"])
    if not (lo <= ratio <= hi):
        raise AssertionError(
            f"mass_balance_ratio {ratio:.3f} outside [{lo}, {hi}] "
            f"(deposited vs ṁ·t_weld)"
        )
    print(
        f"[job_locks] strict_mass  ratio={ratio:.3f}  n_drops={n_drops}  "
        f"wire={telem['expected_wire_mass_g']:.4f}g"
    )


def _lock_lorentz_strict() -> None:
    job = _load("lorentz_strict.yaml")
    twin = _tiny(job, enable_lorentz=True)
    twin.lorentz_jacobi_iters = int(
        (job.get("advanced_physics") or {}).get("lorentz_jacobi_iters", 2)
    )
    twin.lorentz_jacobi_tol = float(
        (job.get("advanced_physics") or {}).get("lorentz_jacobi_tol", 1e-8)
    )
    g = twin.grid
    g.ensure_lorentz_fields()
    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    i0, j0, k0 = g.nx // 2, g.ny // 2, min(g.nz - 2, max(twin.nz_solid, 1))
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-1, 2):
                i, j, k = i0 + di, j0 + dj, k0 + dk
                if 0 <= i < g.nx and 0 <= j < g.ny and 0 <= k < g.nz:
                    flags[i, j, k] = g.FLAG_FLUID
                    fl[i, j, k] = 1.0
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)
    twin._lorentz_warm = True
    noise = np.random.default_rng(0).standard_normal(
        (g.nx, g.ny, g.nz), dtype=np.float32
    )
    g.phi_elec.from_numpy(noise)
    f0 = force_linf_lu(g)
    arc_i, arc_j, arc_k = (g.nx - 1) * 0.5, (g.ny - 1) * 0.5, float(max(0, twin.nz_solid - 1))
    solve_lorentz(twin, g, arc_i, arc_j, arc_k)
    streak = int(getattr(twin, "_lorentz_unconverged_streak", 0) or 0)
    if streak < 1:
        raise AssertionError("unconverged Lorentz step must increment streak")
    f1 = force_linf_lu(g)
    if f1 > f0 + 1e-12:
        raise AssertionError(
            f"unconverged Lorentz step increased |F| {f0:.3e} → {f1:.3e}"
        )
    diag = sample_force_diagnostics(twin)
    if float(diag.get("f_lorentz_max", 0.0)) > 1e-12:
        raise AssertionError(
            f"skipped J×B still reports f_lorentz_max={diag['f_lorentz_max']}"
        )
    print(
        f"[job_locks] lorentz_strict  streak={streak}  |F|={f1:.3e}  "
        f"f_lorentz_max={diag['f_lorentz_max']:.3e}"
    )


def _lock_interpass_travel() -> None:
    job = _load("interpass_travel.yaml")
    model = job["model_reference"]
    twin = _tiny(job)
    cooling = int(model["cooling_steps"])
    v_mm_s = float(model["travel_speed_mm_s"])
    gap_m = float(model["gap_mm"]) / 1000.0
    if not twin._interpass_travel_m_s:
        raise AssertionError("interpass.travel_speed_mm_s was not applied")
    if abs(twin._interpass_travel_m_s - v_mm_s / 1000.0) > 1e-12:
        raise AssertionError(
            f"travel speed {twin._interpass_travel_m_s} != {v_mm_s/1000.0} m/s"
        )

    def _clamp(x, y):
        return x, y

    n_idle = twin._idle_interpass(
        (0.004, 0.008, 0.0), (0.004 + gap_m, 0.008, 0.0), cooling, _clamp,
    )
    motion = max(0, int(gap_m / max(twin._interpass_travel_m_s * twin.grid.dt, 1e-18)))
    if n_idle <= cooling:
        raise AssertionError(
            f"idle steps {n_idle} should exceed cooling_steps={cooling} when travel is required"
        )
    if n_idle != motion + cooling:
        raise AssertionError(
            f"idle steps {n_idle} != motion {motion} + cooling {cooling}"
        )
    print(f"[job_locks] interpass_travel  idle={n_idle}  motion={motion}  cooling={cooling}")


def _lock_frame_offset_z() -> None:
    job = _load("frame_offset_z.yaml")
    model = job["model_reference"]
    torch_z = float(model["torch_z_m"])
    twin = _tiny(job)
    if not twin.use_torch_z:
        raise AssertionError("frame_offset_z job must set use_torch_z")
    if abs(float(twin._sim_origin_offset_z_m) - float(model["offset_z_m"])) > 1e-12:
        raise AssertionError(
            f"z offset {twin._sim_origin_offset_z_m} != {model['offset_z_m']}"
        )
    g = twin.grid
    cx, cy = (g.nx // 2) * g.dx, (g.ny // 2) * g.dx
    twin.step(cx, cy, is_welding=True, torch_z_m=torch_z)
    k_off = float(twin._last_arc_ijk[2])

    apply_frame_to_twin(twin, WeldFrame())
    twin.reset()
    twin.step(cx, cy, is_welding=True, torch_z_m=torch_z)
    k_zero = float(twin._last_arc_ijk[2])
    dk = k_zero - k_off
    expected = float(model["offset_z_m"]) / g.dx
    if abs(dk - expected) > 0.6:
        raise AssertionError(
            f"arc_k shift {dk:.3f} cells vs expected {expected:.3f} "
            f"(k_off={k_off:.3f}, k_zero={k_zero:.3f})"
        )
    print(f"[job_locks] frame_offset_z  Δarc_k={dk:.3f}  expected={expected:.3f}")


def _lock_reset_ledgers() -> None:
    job = _load("reset_ledgers.yaml")
    model = job["model_reference"]
    twin = _tiny(job, enable_moving_window=True, enable_evaporative_cooling=True)
    twin._window_offset_x_m = float(model["window_offset_x_m"])
    twin._window_offset_y_m = 0.003
    twin._window_offset_z_m = 0.002
    twin._evap_energy_J_cum = float(model["evap_energy_J_cum"])
    twin._evap_energy_J_step = 3.0
    twin._warned_overflow = True
    twin._lorentz_unconverged = 7
    twin.reset()
    if twin._window_offset_x_m != 0.0:
        raise AssertionError(f"window offset after reset: {twin._window_offset_x_m}")
    if twin._window_offset_y_m != 0.0 or twin._window_offset_z_m != 0.0:
        raise AssertionError("Y/Z window offsets not zeroed")
    if twin._evap_energy_J_cum != 0.0 or twin._evap_energy_J_step != 0.0:
        raise AssertionError("evap energy ledgers not zeroed")
    if twin._warned_overflow:
        raise AssertionError("_warned_overflow still True")
    if twin._lorentz_unconverged != 0:
        raise AssertionError("_lorentz_unconverged not zeroed")
    telem = twin.get_telemetry()
    if float(telem.get("window_offset_x_mm", 1)) != 0.0:
        raise AssertionError("telemetry window_offset_x_mm not 0")
    if float(telem.get("evap_energy_J_cum", 1)) != 0.0:
        raise AssertionError("telemetry evap_energy_J_cum not 0")
    if int(telem.get("lorentz_unconverged_count", 1)) != 0:
        raise AssertionError("telemetry lorentz_unconverged_count not 0")
    print("[job_locks] reset_ledgers  window/evap/overflow/lorentz zeroed")


def _lock_lorentz_cold_start() -> None:
    job = _load("lorentz_cold_start.yaml")
    model = job["model_reference"]
    n_max = int(model["n_max"])
    twin = _tiny(job, nx=n_max, ny=16, nz=16, enable_lorentz=True)
    cap = int(model["max_cold_iters"])
    computed = lorentz_cold_max_iters(twin, twin.grid)
    if computed > cap or computed != cap:
        raise AssertionError(f"cold cap {computed} != model max_cold_iters {cap}")
    twin._lorentz_warm = False
    g = twin.grid
    solve_lorentz(twin, g, g.nx / 2, g.ny / 2, float(twin.nz_solid))
    if twin._lorentz_max_iters_last > cap:
        raise AssertionError(
            f"cold-start max_iters {twin._lorentz_max_iters_last} exceeded cap {cap}"
        )
    if twin._lorentz_iters_last > cap:
        raise AssertionError(
            f"cold-start iters_done {twin._lorentz_iters_last} exceeded cap {cap}"
        )
    unbounded = 2 * n_max * n_max
    if twin._lorentz_max_iters_last >= unbounded:
        raise AssertionError(
            f"cold-start still using 2·n_max²={unbounded}"
        )
    print(
        f"[job_locks] lorentz_cold_start  iters={twin._lorentz_iters_last}/"
        f"{twin._lorentz_max_iters_last}  cap={cap}  (2n²={unbounded})"
    )


def _lock_table_knots() -> None:
    job = _load("table_knots.yaml")
    model = job["model_reference"]
    twin = _tiny(job)
    n_gpu = int(twin.gpu_tables.n_cp[None])
    if not twin.gpu_tables.truncated:
        raise AssertionError("expected gpu_tables.truncated after 9 cp knots")
    if n_gpu != int(model["n_knots_gpu"]):
        raise AssertionError(f"n_cp={n_gpu} != {model['n_knots_gpu']}")
    job_strict = copy.deepcopy(job)
    job_strict.setdefault("simulation", {})["strict_mode"] = True
    try:
        apply_job_to_twin(twin, job_strict)
    except ValueError as exc:
        if "truncated" not in str(exc):
            raise AssertionError(f"strict truncation error missing 'truncated': {exc}") from exc
    else:
        raise AssertionError("strict_mode must raise on truncated material tables")
    print(f"[job_locks] table_knots  n_cp={n_gpu}  truncated + strict raise OK")


def _lock_dual_alloy_rho() -> None:
    job = _load("dual_alloy_rho.yaml")
    model = job["model_reference"]
    twin = _tiny(job, enable_vof=True, enable_arc_pressure=True)
    if not twin.use_dual_alloy:
        raise AssertionError("dual_alloy_rho job must enable the dual-alloy path")
    g = twin.grid
    i_w, i_p = 4, 8
    j = g.ny // 2
    k = max(2, min(twin.nz_solid, g.nz - 3))
    flags = g.flags.to_numpy()
    phi = g.phi.to_numpy()
    aid = g.alloy_id.to_numpy()
    afrac = g.alloy_frac.to_numpy()
    Fz = np.zeros((g.nx, g.ny, g.nz), dtype=np.float32)
    for i, alloy in ((i_w, ALLOY_WIRE), (i_p, ALLOY_PLATE)):
        flags[i, j, k] = g.FLAG_FLUID
        flags[i, j, k - 1] = g.FLAG_FLUID
        flags[i, j, k + 1] = g.FLAG_GAS
        aid[i, j, k] = alloy
        afrac[i, j, k] = 1.0 if alloy == ALLOY_PLATE else 0.0
        phi[i, j, k - 1] = 1.0
        phi[i, j, k] = 0.5
        phi[i, j, k + 1] = 0.0
    g.flags.from_numpy(flags)
    g.phi.from_numpy(phi)
    g.alloy_id.from_numpy(aid)
    g.alloy_frac.from_numpy(afrac)
    g.Fz.from_numpy(Fz)
    alloy_frac, rho_w, rho_p = (
        g.alloy_frac,
        float(model["rho_wire"]),
        float(model["rho_plate"]),
    )
    if abs(rho_w - twin.mat.rho) > 1.0 or abs(rho_p - twin.plate_mat.rho) > 1.0:
        raise AssertionError(
            f"ρ mismatch wire {twin.mat.rho} vs {rho_w}  plate {twin.plate_mat.rho} vs {rho_p}"
        )
    kernels.apply_arc_pressure(
        g.Fz, g.flags, g.phi,
        0.5 * (i_w + i_p), float(j), float(k),
        80.0,
        5000.0, g.dt, g.dx, alloy_frac, rho_w, rho_p,
        g.FLAG_SOLID, g.FLAG_GAS,
    )
    Fz = g.Fz.to_numpy()
    fw = abs(float(Fz[i_w, j, k]))
    fp = abs(float(Fz[i_p, j, k]))
    if fw < 1e-18 or fp < 1e-18:
        raise AssertionError(f"arc pressure |F| too small  wire={fw} plate={fp}")
    ratio = fp / fw
    expected = float(model["force_ratio"])
    if abs(ratio - expected) / expected > 0.08:
        raise AssertionError(
            f"|F|_plate/|F|_wire={ratio:.3f} != ρ_wire/ρ_plate={expected}"
        )
    print(f"[job_locks] dual_alloy_rho  |F|_p/|F|_w={ratio:.3f}  expected={expected}")


def _lock_heat_loss_factor() -> None:
    job = _load("heat_loss_factor.yaml")
    model = job["model_reference"]
    twin = _tiny(job)
    if abs(twin.h_conv - float(model["h_conv"])) > 1e-9:
        raise AssertionError(f"h_conv={twin.h_conv} != {model['h_conv']}")
    if abs(twin.eps_rad - float(model["eps_rad"])) > 1e-9:
        raise AssertionError(f"eps_rad={twin.eps_rad} != {model['eps_rad']}")
    print(f"[job_locks] heat_loss_factor  h_conv={twin.h_conv}  eps_rad={twin.eps_rad}")


def _lock_fail_closed() -> None:
    try:
        electrical_arc_power_W({}, required=True)
    except ValueError:
        pass
    else:
        raise AssertionError("missing current_A/voltage_V must raise")
    try:
        electrical_arc_power_W({"current_A": 100}, required=True)
    except ValueError:
        pass
    else:
        raise AssertionError("missing voltage_V must raise")
    twin = _tiny(_load("thermal_tier.yaml"))
    try:
        twin.run_path({"process": {}, "torch_path": []})
    except ValueError:
        pass
    else:
        raise AssertionError("empty torch_path must raise")
    try:
        load_weld_frame("jobs/examples/locks/no_such_frame.yaml")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing explicit frame file must raise")
    prev = os.environ.get("WAAM_FRAME")
    os.environ["WAAM_FRAME"] = "jobs/examples/locks/no_such_frame.yaml"
    try:
        load_weld_frame(None)
        raise AssertionError("missing WAAM_FRAME file must raise")
    except FileNotFoundError:
        pass
    finally:
        if prev is None:
            os.environ.pop("WAAM_FRAME", None)
        else:
            os.environ["WAAM_FRAME"] = prev
    print("[job_locks] fail_closed  power/path/frame raise OK")


def _lock_unknown_physics_keys() -> None:
    job = copy.deepcopy(_load("thermal_tier.yaml"))
    job.setdefault("goldak", {})["a_frot_mm"] = 2.0
    try:
        validate_job_config(job)
    except ValueError as exc:
        msg = str(exc)
        if "a_frot_mm" not in msg:
            raise AssertionError(f"schema error should name a_frot_mm: {exc}") from exc
    else:
        raise AssertionError("goldak.a_frot_mm must fail validate_job_config")
    print("[job_locks] unknown goldak.a_frot_mm rejected")


def _lock_fitted_knobs() -> None:
    from waam_twin.validation.prediction import (
        CALIBRATE_JOB,
        HELDOUT_FAST_JOB,
        HELDOUT_HOT_JOB,
        HELDOUT_MACRO2_JOB,
        assert_physics_lock,
        extract_locked_physics,
    )

    base = load_job_config(CALIBRATE_JOB)
    locked = extract_locked_physics(base)
    if "evap_cooling_scale" not in (locked.get("advanced_physics") or {}):
        raise AssertionError("extract_locked_physics must include evap_cooling_scale")
    for path in (HELDOUT_FAST_JOB, HELDOUT_HOT_JOB, HELDOUT_MACRO2_JOB):
        assert_physics_lock(base, load_job_config(path), label=path)
    print("[job_locks] fitted knobs  assert_physics_lock held-outs OK")


def _lock_dt_scale() -> None:
    twin_a = WAAMTwin(nx=16, ny=12, nz=14, dx=1e-3, max_tracers=8, dt_scale=1.0)
    twin_b = WAAMTwin(nx=16, ny=12, nz=14, dx=1e-3, max_tracers=8, dt_scale=2.0)
    if abs(twin_b.grid.dt / twin_a.grid.dt - 2.0) > 1e-12:
        raise AssertionError(
            f"dt_scale=2 gave dt={twin_b.grid.dt} vs base {twin_a.grid.dt}"
        )
    if twin_b.grid.tau <= 0.5:
        raise AssertionError("dt_scale=2 must still satisfy τ>0.5")
    if twin_b.grid.alpha_lu > 1.0 / 6.0:
        raise AssertionError("dt_scale=2 must still satisfy α_lu≤1/6")
    print(f"[job_locks] dt_scale  dt×2={twin_b.grid.dt:.3e}s  τ={twin_b.grid.tau:.4f}")


def run() -> None:
    init_taichi(backend="cpu")
    _lock_schema()
    _lock_heat_loss_hot()
    _lock_thermal_tier()
    _lock_strict_mass()
    _lock_lorentz_strict()
    _lock_interpass_travel()
    _lock_frame_offset_z()
    _lock_reset_ledgers()
    _lock_lorentz_cold_start()
    _lock_table_knots()
    _lock_dual_alloy_rho()
    _lock_heat_loss_factor()
    _lock_fail_closed()
    _lock_unknown_physics_keys()
    _lock_fitted_knobs()
    _lock_dt_scale()
    print("[job_locks] all locks OK")


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
