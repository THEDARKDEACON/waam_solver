# Handoff: free-surface LBM, LES and Park recalibration (shop mesh)

Status as of 2026-10-22. Nothing is committed; all work is uncommitted in the
`waam_twin/` git repo (46 files changed, most from earlier sessions in this
same campaign). `validation/test_fslbm.py` is new and untracked.

## 1. Environment and editing rules

- Quadrants 1.3.3 (Taichi fork, `import quadrants as qd`, used as `ti`).
  **Kernels do not accept default arguments**; every call site must pass all args.
- GPU: RTX 3050 **4 GB**. Only **one** full-size run fits at a time; two in
  parallel die with `CUDA_ERROR_OUT_OF_MEMORY`.
- Python 3.13. Env for every run (run from `E:\home\gareth-joel\Downloads\WAAM`):
  `PYTHONPATH=E:\home\gareth-joel\Downloads\WAAM`, `WAAM_ALLOW_ABSOLUTE_PATHS=1`,
  `PYTHONIOENCODING=utf-8`. Tests: `python -m waam_twin.validation.test_<name>`.
- Direct editor writes to `E:` were blocked in this session. Edits were made with
  patch scripts in `D:\ansys_2024r2\` using `_patchlib.py`:
  `apply(rel_path, [(old, new), ...])` (each `old` must match exactly once; CRLF
  preserved) and `write(rel_path, content)`. Paths are relative to `waam_twin/`.
  `D:\ansys_2024r2\_fslbm_src.py` is the staged master copy of `kernels/fslbm.py`
  (edit it, then `write("kernels/fslbm.py", ...)`), or just edit the repo file
  directly if your tooling can.

## 2. Mesh / time step in use

Shop tier (`jobs/examples/bead_calibrate.yaml`): **dx = 0.4 mm**, domain
80 x 40 x 25 mm -> 200 x 100 x 63 cells (~1.26 M), plate 9 mm thick,
**dt = 4.0e-5 s** (50 000 steps = 2.0 s). Liquid-steel viscosity puts LBM tau
near 0.5 at this dx; this is why the Mach clamp used to saturate. The
Smagorinsky model (below) is what removes the dependence on the clamp, so the
same approach carries to finer meshes (HPC tier, dx 0.2-0.3) without
re-stabilising by hand. Typical cost: 5.7-6.5 ms/step, ~6-7 min per 2.5 s run.

## 3. What was done (the user's "4 steps")

1. **Smagorinsky LES** (Hou et al. form in the TRT/variable-tau collision).
   `twin.smagorinsky_cs` default 0.17. On bead_calibrate: Mach-clamp hits
   ~6900/10000 steps -> 0; u_max 0.1-0.26 m/s; energy ledger closes. Flow
   tests (Poiseuille 0.51%, cavity, etc.) pass.
2. **Free-surface LBM (Koerner/Thuerey)** in `kernels/fslbm.py`, driven by
   `_free_surface_stream()` in `solvers/coupled_step.py`; enabled when
   `enable_vof` and `simulation.use_free_surface_lbm: true` (job key is now
   parsed in `job.py`; previously it was silently ignored).
   - Gas cells hold no populations; interface reconstruction
     `f_q = feq_q(rho_G,u) + feq_qbar(rho_G,u) - f*_qbar`, rho_G = 1 + 3*gamma*kappa
     (Brackbill curvature, clamp +-0.5).
   - Mass lives in `phi` between steps; `g.fs_mass`, `fs_mark`, `fs_energy`,
     `fs_mass_lost`, `fs_n_surface`, `fs_share` allocated by
     `grid.ensure_fslbm_fields()`.
   - **Fill-weighted energy**: cell thermal energy is w*H*V with w = clamp(phi,0,1)
     in FS mode (w = 1 otherwise). Enthalpy is carried upwind with the mass flux.
     All sources/sinks book w-weighted energy; `physics/ledger.py` stores sum(w*H*V).
     Thermal kernels take trailing `(phi, fs_iface)`; pass `g.phi, twin.fs_iface_flag()`
     (or `g.phi, -1` when FS is off, e.g. `thermal_only.py` and the thermal tests).
   - **Buried interface cells** (no gas neighbour) are converted to fluid
     regardless of mass; their (negative) excess goes to `fs_mass_lost` and is
     spread evenly over surface interface cells each step by
     `fs_count_surface` + `fs_settle_unplaced` (energy moved at each surface
     cell's own H, booked to `energy_acc[3]`). Without this the bead filled with
     permanent voids that insulated it from the plate.
   - **Deposition**: `fs_deposit_into_interface` tops up partly filled interface
     cells in the droplet footprint before `feed_wire_surface` claims gas cells
     (shared volume budget; energy (1-eps)*H_drop matches the droplet ledger).
   - **Surface losses** (convection, radiation, evaporation) remove q*A*dt per
     exposed face regardless of fill: dH divided by max(w, 0.25)
     (`_LOSS_MIN_FILL` in `kernels/thermal.py`).
   - `validation/test_fslbm.py`: dam break (mass 1.3e-5, energy 2e-8),
     flat pool (u_max 4e-9), Laplace droplet (ratio 0.967). Passes.
3. **Vapor cap energy (~13%)**: explained. With LES the cap removes 4.1 J (was
   134.7 J); evaporation 28.6 J. The cap was a symptom of clamp-saturated flow
   not carrying heat away, not a leak.
4. **Recalibration** (see section 4). No knob values changed.

Other changes in this campaign: `lorentz_solve_every_s` job key (schema +
parser), `use_free_surface_lbm` added to the physics-lock fingerprint in
`validation/prediction.py` (held-outs must match the calibrate job's solver).

## 4. Calibration state (targets: Park et al. 2019, flat P-GMAW, etched)

Targets W 6.1 / D 1.8 / H 2.2 mm (`reference:` in bead_calibrate.yaml). These
are lab measurements; Ansys was only used for thermal-conduction parity (it has
no melt-pool flow, so it cannot give bead shape).

Protocol: `D:\ansys_2024r2\_cal_ss.py JOB STEPS FS X0 X1 [eta= a_front= a_rear= b= c= recoil= evap= every=]`
(strict mode off, Lorentz solved every 2.5e-4 s). Steady cross-section W/D from
the liquidus T_max contour per slice, H from the phi crown. Use **62 500 steps
(2.5 s) and slices 22-26 mm**; at 2.0 s those slices are still half liquid and H
reads high. The old 8000-step metric is meaningless: the path starts at the
plate edge (x = 15 mm) and the pool has not formed/frozen.

| Run (FS on, 2.5 s unless noted)              | W mm (err)   | D mm (err)  | H mm (err)  | frozen |
|----------------------------------------------|--------------|-------------|-------------|--------|
| Before void fix, 2.0 s, baseline knobs        | 8.5 (+40%)   | 1.5 (-18%)  | 3.1 (+40%)  | 0.01   |
| **Baseline knobs (eta 0.82, Goldak 2.0/3.8/2.8/2.6)** | **5.36 (-12%)** | **1.83 (+2%)** | **2.27 (+3%)** | 0.87 |
| eta 0.90                                      | 5.65 (-7%)   | 1.99 (+10%) | 2.55 (+16%) | 0.79   |
| b 3.2 mm (2.0 s)                              | 5.48 (-10%)  | 1.70 (-6%)  | 2.82 (+28%) | 0.47   |

Locked: baseline knobs (unchanged values). Width is the only off metric and is
inside the job's own stated band (+-15-25%). Heat input raises D faster than W;
source half-width b barely moves W. Untested next lever: wider + shallower
source together (e.g. eta 0.90, b 3.6, c 2.0), or wetting/Marangoni (those are
`knobs_locked`, so changing them is a policy decision for the user).

Job changes: `use_free_surface_lbm: true` and `lorentz_solve_every_s: 2.5e-4`
in bead_calibrate + heldout_fast/hot/macro2; `model_reference.n_steps` 62500 and
note updated in bead_calibrate. HPC jobs were **not** switched or refit.

## 5. Closed on 2026-10-22 (follow-up session)

1. Held-out trend check, done. bead_calibrate_heldout_fast (15 mm/s, same knobs, 62 500 steps). Slices 22-26 mm are the startup ramp at this speed (H 1.24 mm). The steady frozen stretch is x = 30-36 mm: W 4.98, D 1.31, H 1.80, frozen 0.98. Against the 10 mm/s lock (W 5.36, D 1.83, H 2.27) every dimension is down. The W*H proxy is 8.94 vs 12.17, ratio 0.73 against the 2/3 mass-balance expectation (travel 1.5x, same wire feed). W is fusion width, so W*H is not the deposited area. Mach-clamp hits varied between the two identical runs (531 and 2877 of 62 500).
2. Width tuning skipped. The lock is inside the job's own +-15-25% band and the held-out moved the right way. eta and Goldak are unchanged.
3. Test list passed (exit 0): thermal_diffusion, cooldown_energy, physics_tier, dual_alloy, probe_recorder, job_locks, lorentz_physical_scale, intensive_lorentz_convergence, advanced_weld_forces, force_direction_gate, intensive_coupled_physics, lbm_cavity, lbm_poiseuille (0.51%), mass_conservation, lbm_propagation, enthalpy_cap, enthalpy_cap_init, wetting_no_volume, mrt_force_mass, fslbm, heated_cavity, stefan_solidification (7.42%). test_ledger_host and test_measure_geometry are pytest functions with no __main__; their functions were called directly and passed. Logs in D:\ansys_2024r2\_tests\. Not run, and still 62 500 steps if launched: test_calibrated_pool, test_pool_geometry, test_bead_macrograph_gate, test_heldout_prediction.
4. Diff and calibration summary delivered in chat. Nothing committed.

_cal_ss.py now prints absolute W/D/H when the job reference is null, plus the W*H proxy.

## 6. Known risks / gotchas

- `strict_mode: true` in the jobs: the Lorentz solver showed nondeterministic
  strict-mode non-convergence earlier; calibration runs used strict off.
- Mach clamp still fires on ~0.1-0.4% of steps in FS mode (u ~0.75 m/s peaks).
- Dam-break test books ~6% of E through `energy_acc[3]` (relabel) with an
  extreme H gradient; closure is still exact, but relabel is not physical heat.
- `_cal_ss.py` prints its PROFILE block twice (duplicated code, harmless).
- Diagnostics on D: `_ledger_full.py JOB N`, `_fs_diag.py JOB N`,
  `_fs_geom.py`, `_fs_slices.py`, `_fs_laplace.py`.
- Security: process PID 11740, `pythonw.exe` running `version8.txt` from
  `AppData\Local\Microsoft\EdgeUpdate\{BBD28F4B-...}\1.3.348.61\`, looks like a
  malware persistence pattern. Not touched; flagged to the user.
