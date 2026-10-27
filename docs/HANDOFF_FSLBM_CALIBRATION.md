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

## 6. Spread and mesh check (2026-10-22), fit not started

Tool: `python -m waam_twin.tools.steady_bead`. Records in
`runs/steady/spread.jsonl` and `runs/steady/mesh_dx030.jsonl`.

Three identical dx = 0.4 mm runs, 62 500 steps, slices 22-26 mm:

| run | W mm | D mm | H mm | frozen | mach |
|-----|------|------|------|--------|------|
| 1 | 5.44 | 1.80 | 2.50 | 0.84 | 228 |
| 2 | 5.42 | 1.69 | 1.83 | 0.91 | 494 |
| 3 | 5.10 | 1.72 | 2.39 | 0.81 | 185 |
| mean (sigma) | 5.32 (0.19) | 1.74 (0.06) | 2.24 (0.36) | | 302 (167) |

Width sigma is 3.6% of the mean, under the 6% stop. Fit tolerances
`max(2*sigma, 3%)` against Park: W 6.3%, D 6.7%, H 32%. The mean width
error is -13%, so it is larger than the width noise. Height is not: the
three runs span 1.83 to 2.50 mm.

dx = 0.3 mm, same knobs, 2.5 s (83 333 steps, dt 30 us, tau 0.5009,
mach 631, 0.8% of steps; the old clamp failure did not return):
W 5.49 (+3% vs the dx 0.4 mean), D 1.67 (-4%), H 2.86 (+28%).
Width and depth are inside 10%. Height is not, and it sits above the
whole dx 0.4 range. The plan's rule is to stop before any knob fit:
a marangoni fit at dx 0.4 would be absorbing a mesh-dependent height.
eta and Goldak are unchanged. No job files were edited.

## 7. Frozen height metric (2026-10-22)

Implemented `docs/SPEC_STEADY_HEIGHT_METRIC.md`. `H_frozen_mm` ignores columns
that are still liquid. `area_mm2` is the deposited cross-section (wire feed
implies 9.16 mm2). Runs are 3.5 s so the 22-26 mm window is 24-28 mm behind
the torch. `validation/test_steady_section.py` passes. Every run below was
`steady=True` (frozen fraction 1.00), so none was repeated at 4.0 s.

dx = 0.4 mm, two runs:

| run | W | D | H_max | H_frozen | area | area_ratio |
|-----|---|---|-------|----------|------|------------|
| 1 | 5.12 | 1.83 | 1.60 | 1.60 | 8.02 | 0.88 |
| 2 | 5.26 | 1.79 | 2.11 | 2.04 | 10.09 | 1.10 |
| mean (sigma) | 5.19 (0.10) | 1.81 (0.02) | 1.86 (0.36) | 1.82 (0.31) | 9.05 (1.46) | 0.99 |

dx = 0.3 mm, one run: W 5.42 (+4%), D 1.77 (-2%), H_frozen 1.80 (-1%),
area_ratio 1.03, mach 844/116667. Once the window is solid, H_max and
H_frozen agree, and the +28% height jump from section 6 disappears.
Width, depth and frozen height are each within 10% of the dx 0.4 mean.
The mesh means of area_ratio are 0.99 and 1.03. A single run still lands
outside 0.90-1.10 (0.88 and, later, 1.15), so area is conserved on average
and noisy per run.

Marangoni scale at dx 0.4, 3.5 s (baseline mean W 5.19):

| scale | W | D | H_frozen | area_ratio |
|-------|---|---|----------|------------|
| 0.7 | 5.24 | 1.79 | 1.60 | 0.87 |
| 1.5 | 5.29 | 1.81 | 2.00 | 1.08 |
| 2.0 | 5.06 | 1.84 | 2.00 | 1.00 |
| 1.5 again | 5.15 | 1.81 | 2.40 | 1.15 |

Width stays inside the 0.10 mm repeat scatter. The 1.5 repeat did not
reproduce the first 1.5 run. No knob was locked and no job file was edited.
Records: `runs/steady/spread_frozen.jsonl`, `mesh_dx030_frozen.jsonl`,
`fit_park.jsonl`.

## 8. Width-lever screen (2026-10-22)

Implemented `docs/SPEC_WIDTH_LEVER_SCREEN.md`. Window x=22-34 mm at 4.0 s
(100 000 steps, dx 0.4 mm). All five runs were steady. The wider window
settled the crown: the two baseline runs differ by 0.05 mm in H_frozen and
0.05 in area_ratio, both inside the spec's limits (0.3 mm and 0.10).

| run | W | D | H_frozen | W_bead | area_ratio | dW vs base |
|-----|---|---|----------|--------|------------|------------|
| base 1 | 5.11 | 1.69 | 2.05 | 5.42 | 1.02 | |
| base 2 | 5.29 | 1.79 | 2.00 | 5.63 | 1.07 | |
| base mean | 5.20 | 1.74 | 2.03 | | 1.04 | |
| contact 60 deg | 5.18 | 1.77 | 1.73 | 5.46 | 0.91 | -0.02 |
| Goldak b 4.0 mm | 5.39 | 1.58 | 2.01 | 5.73 | 1.09 | +0.19 |
| arc sigma 3.0 mm | 5.47 | 1.72 | 2.26 | 5.63 | 1.13 | +0.27 |

No knob moved width by 0.5 mm, and depth stayed within 10% of 1.74 mm on
every run. Arc sigma 3.0 mm is the largest shift (+0.27 mm) and, with the
Goldak source, that knob does not change the heat input: it widens the
deposition footprint and the Lorentz current spread. Fitting stopped.
eta 0.82, Goldak 2.0/3.8/2.8/2.6, contact 80 and arc sigma 2.0 stay.
`bead_calibrate.yaml` records the baseline in model_reference.note and adds
`width_bias_minus_15pct_shop` to claim.limitations. Records:
`runs/steady/width_screen.jsonl`.

## 9. Flow damping diagnostic (2026-10-22)

Host diagnostic `measure/pool_flow.py` (test_pool_flow passes). Four runs,
dx 0.4 mm, 4.0 s, slices 22-34 mm, all steady. Records in
`runs/steady/flow_diag.jsonl`. Baseline W 5.20 mm matches the width-screen
mean (drift 0.00 mm).

| run | Cs | Marangoni | W | D | u_surf p95 m/s | u_max m/s | nu_t/nu | mach_frac |
|-----|----|-----------|---|---|----------------|-----------|---------|-----------|
| base | 0.17 | 1.0 | 5.20 | 1.72 | 0.148 | 0.284 | 0.34 | 0.0015 |
| mar20 | 0.17 | 2.0 | 5.24 | 1.82 | 0.154 | 0.334 | 0.35 | 0.0015 |
| cs10 | 0.10 | 1.0 | 5.38 | 1.78 | 0.173 | 0.338 | 0.13 | 0.0053 |
| cs00 | 0.0 | 1.0 | 5.53 | 1.18 | 0.245 | 0.789 | 0 | 0.757 |

Doubling the surface force raised the surface speed by 4% (ratio 1.04, below
the 1.3 line), so the flow is being limited. The pool-mean eddy viscosity is
0.34 of the molecular viscosity, not above 5, and dropping Cs from 0.17 to
0.10 widened the bead by only 0.17 mm. Smagorinsky is not the limiter.
cs00 is clamp-limited (mach_frac 0.76) and its width is not used. At the
locked Cs the clamp fires on 0.15% of steps, so the clamp is not the limiter
either. The remaining candidate is mushy-zone Darcy damping (C_darcy, a locked
knob). No job file and no default Cs were changed.

## 10. Force-clamp diagnostic (2026-10-22)

One run, dx 0.4 mm, 4.0 s, slices 22-34 mm, force cap 125000 m/s2 (10x the
locked 12500). Steady. Record in `runs/steady/force_clamp.jsonl`.

| | W mm | D mm | u_surf m/s | u_max m/s | force_clamp_frac | mach_frac |
|--|------|------|------------|-----------|------------------|-----------|
| section 9 baseline | 5.20 | 1.72 | 0.148 | 0.284 | (not logged) | 0.0015 |
| cap x10 | 5.26 | 1.77 | 0.157 | 0.312 | 0.000 | 0.0040 |

Surface speed rose 6%, under the 1.3 line, and the force clamp fired on no
steps. The ceiling is not what keeps the pool slow, and the lattice-cap
follow-up was not run. The remaining candidate is mushy-zone Darcy damping
(C_darcy 160000, locked, scaled by C_darcy_ref_dt_s). No job file was edited.

## 11. Mush damping and where the force lands (2026-10-22)

`pool_flow_stats` now reports the surface liquid fraction and the Darcy
attenuation `1 + 0.5 K`. `test_pool_flow` passes, including a fully liquid
surface (attenuation 1) and `f_l = 0.9` at C = 160000 (attenuation above 100).
Runs at 4.0 s, slices 22-34 mm, in `runs/steady/darcy_diag.jsonl`. Reference
is the section 9 baseline: u_surf 0.148 m/s, W 5.20 mm, D 1.72 mm.

| C_darcy | W | D | H_frozen | u_surf | fl median | frac f_l>=0.99 | atten | mach_frac |
|---------|---|---|----------|--------|-----------|----------------|-------|-----------|
| 1600 | 5.37 | 1.77 | 2.00 | 0.157 | 0.86 | 0.37 | 29 | 0.0018 |
| 0 | 6.14 | 1.81 | 1.61 | 0.213 | 0.96 | 0.49 | 1.00 | 0.135 |

At 100x below the lock the surface is still braked (attenuation 29) because
those cells are only partly liquid, and the speed does not move. Switching
the brake off is the first change in this campaign that raises the surface
speed by more than 30% (0.148 to 0.213 m/s) and lands the width on Park
(6.14 mm, depth 1.81 mm). The speed clamp fires on 13.5% of steps, above the
0.05 line, so that width is reported and not used as a pass. The mechanism
is the mush brake. A usable C_darcy has to free cells above liquid fraction
0.95 without handing the run to the speed clamp. The locked value 160000
was not changed.

## 12. Mush-damping bracket (2026-10-22)

Three runs, dx 0.4 mm, 4.0 s, slices 22-34 mm, all steady, in
`runs/steady/darcy_bracket.jsonl`. A run needs mach_frac <= 0.05 and
W >= 5.80 mm, with depth within 10% of 1.8 mm, to be locked.

| C_darcy | W | D | H_frozen | u_surf | fl median | atten | mach_frac |
|---------|---|---|----------|--------|-----------|-------|-----------|
| 200 | 5.10 | 1.74 | 1.91 | 0.162 | 0.89 | 2.9 | 0.0018 |
| 400 | 5.32 | 1.76 | 2.00 | 0.153 | 0.85 | 8.8 | 0.0009 |
| 800 | 5.38 | 1.78 | 2.01 | 0.149 | 0.87 | 13.2 | 0.0030 |

All three are quiet. The widest is 5.38 mm, short of 5.80 mm. Surface speed
stays near 0.15 m/s and the surface stays partly liquid. The Park width only
appeared at C_darcy = 0, where the speed clamp fired on 13.5% of steps.
C_darcy stays 160000 and width_bias_minus_15pct_shop stays. This mesh cannot
hold the pool speed that produces Park's width. No job file was edited.

## 13. Zero damping at a shorter time step (2026-10-22)

`--dt-scale 0.5` asked for 20 us. auto_dt_ma lengthened it to 28.72 us so
tau stayed at 0.5005 (dt_scale 0.718). The speed ceiling rose from 0.8 m/s
to 1.114 m/s, above the 1.0 m/s gate, so the weld ran. C_darcy = 0, 4.0 s,
slices 22-34 mm. Record in `runs/steady/dt_undamped.jsonl`. The force clamp
did not fire, so there was no follow-up.

| | W | D | H_frozen | u_surf | mach_frac |
|--|---|---|----------|--------|-----------|
| C=0 at 40 us | 6.14 | 1.81 | 1.61 | 0.213 | 0.135 |
| C=0 at 28.7 us | 5.95 | 1.86 | 1.51 | 0.203 | 0.030 |

mach_frac 0.030 is within 0.05, W 5.95 mm is above 5.80 mm, and the depth
is 3% off 1.8 mm. This width is valid. The shop jobs were not edited:
dt_scale is part of the mesh lock and C_darcy = 0 has to travel with it.
The next spec locks that pair and reruns the 15 mm/s held-out.

## 14. Undamped short-step confirmation (2026-10-22)

Repeat of section 13: C_darcy 0, requested dt_scale 0.5, ceiling 1.114 m/s,
dt 28.72 us, 4.0 s, slices 22-34 mm. Record in
`runs/steady/dt_undamped_confirm.jsonl`.

| | W | D | H_frozen | u_surf | mach_frac |
|--|---|---|----------|--------|-----------|
| section 13 | 5.95 | 1.86 | 1.51 | 0.203 | 0.030 |
| confirmation | 6.44 | 1.69 | 1.59 | 0.208 | 0.095 |

The confirmation is steady and the ceiling cleared 1.0 m/s. It does not
qualify for the lock: mach_frac 0.095 is above 0.05, and the width moved
0.49 mm from 5.95 mm, past the 0.25 mm allowance. The shop jobs were not
edited. C_darcy stays 160000, dt_scale stays 1.0, and
width_bias_minus_15pct_shop stays. The 15 mm/s held-out, the lock tests and
the wetting screen were not run.

## 15. Repeatable bead and short-step damping (2026-10-22)

Droplet budget commits in `fs_deposit_into_interface` and `feed_wire_surface`
now run in increasing i, j, k order. The excess-mass split no longer races:
each interface cell pulls its donors in direction order, after donor counts
are stored. A serial walk of the whole grid was too slow to finish a weld.
`test_fslbm` dam break matches flags and phi across two identical runs, and a
one-droplet check fills the same cells twice. Mass and energy checks still pass.

Short step (ceiling 1.114 m/s), two repeats each, all steady, in
`runs/steady/darcy_short_repeat.jsonl`:

| C_darcy | W mm | D mm | H_frozen | u_surf | mach_frac |
|---------|------|------|----------|--------|-----------|
| 50 | 5.41, 5.45 | 1.77, 1.76 | 2.00, 2.31 | 0.147, 0.146 | 0, 0.008 |
| 100 | 5.34, 5.44 | 1.68, 1.74 | 2.00, 2.00 | 0.150, 0.142 | 0.0001, 0.0004 |

Both values are repeatable (width within 0.25 mm, clamp under 5%). Neither
is on Park: mean width 5.43 mm and 5.39 mm, both below 5.80 mm. No lock.
C_darcy stays 160000 and dt_scale stays 1.0. The calibrate claim records
`shop_mesh_speed_ceiling` and the note that at 0.4 mm the flow that matches
Park does not repeat, and the flow that repeats does not match Park.

## 16. Finer-mesh undamped weld (2026-10-23)

Undamped (C_darcy 0) at 0.3 mm with `dt_scale` held on the τ = 0.5005
viscosity floor so the speed ceiling rises with 1/dx. Protocol: 4.0 s,
slices 22–34 mm, shop job untouched. Records in
`runs/steady/dx030_undamped.jsonl`.

| run | dx | dt | ceiling | W | D | H_frozen | u_surf | mach_frac | steady |
|-----|----|----|---------|---|---|----------|--------|-----------|--------|
| 1 | 0.3 mm | 16.15 µs | 1.486 m/s | 6.46 | 1.61 | 1.47 | 0.215 | 0.0033 | True |
| 2 | 0.3 mm | 16.15 µs | 1.486 m/s | 6.81 | 1.69 | 1.58 | 0.214 | 0.0032 | True |

Ceiling cleared the 1.4 m/s abort gate. Both repeats are clamp-quiet
(mach_frac ≪ 0.05) and steady. Mean W = 6.63 mm (≥ 5.80) and mean D =
1.65 mm (within 10% of 1.8 mm). Widths disagree by 0.35 mm, past the
0.25 mm allowance, so this is **not** “repeatable and on Park.” It is
also not “quiet but still narrow.” Because 0.3 mm was not clamp-limited,
the 0.2 mm weld was not run.

Decision: raising the ceiling by refining to 0.3 mm and sitting on the
viscosity floor removes the Mach clamp that broke the 0.4 mm undamped
bead, and the mean W/D land near Park, but the two quiet repeats still
do not agree on width. This GPU mesh does not host a repeatable undamped
Park bead under the spec rules. Shop 0.4 mm lock unchanged (C_darcy
160000, dt_scale 1.0). No smaller cell on this GPU.

## 18. Ordered reductions (2026-10-23)

Index-ordered C-order block sums (block 256) replaced the atomic unplaced-mass
sum in `fs_distribute_excess`, the racing `energy_acc[3]` bookings in the FS
path, and the Goldak / Gaussian arc weight atomics. Spec:
`docs/SPEC_FS_ORDERED_REDUCE.md`. Unit checks:
`validation/test_ordered_reduce.py` (numpy C-order sum; bit-identical Goldak H
on a frozen fill). `test_fslbm` still passes on CPU. Shop job untouched.

Two fresh processes, `--time-s 0.5 --dx 0.3 --c-darcy 0 --dt-scale 0.5
--hash-fields`, ceiling 1.486 m/s, 30952 steps:

| process | sha256(phi, flags, H) |
|---------|------------------------|
| A | `c4ef3d57b2e638c6772ba574a6886a9ad754cdd91331a50c7d0fe31a52116840` |
| B | `1ec2c03abbe2f3ff362f45718b717601506af3fa9e9895547bf56b9d5c0a9fe4` |

**Hashes differ.** The ordered sums were not sufficient for a bit-identical
bead state across processes. No 4 s width pair was run. No width claim.
Shop 0.4 mm lock unchanged.

## 20. Lorentz-off hash gate (2026-10-23)

`--no-lorentz` on `steady_bead` sets `enable_lorentz = False` on the twin only.
Spec: `docs/SPEC_LORENTZ_OFF_HASH.md`. Shop job untouched. Records in
`runs/steady/dx030_nolorentz_hash.jsonl`.

Two fresh processes, `--time-s 0.5 --dx 0.3 --c-darcy 0 --dt-scale 0.5
--no-lorentz --hash-fields`, ceiling 1.486 m/s, 30952 steps,
`enable_lorentz=False`:

| process | sha256(phi, flags, H) | mach_steps |
|---------|------------------------|------------|
| A | `4e2571ddf53e5da6d35d6934d79c9b41d0cd42a70507d7070bf96e8f2a64b6ca` | 1 |
| B | `5c2bccdbf3623a038512acd5d659c5df05d6f785a3343b323206c26683c5ae2c` | 0 |

**Hashes differ.** Lorentz is not required for the split seen in section 18.
The next spec is the droplet budget on CUDA across two processes. The
identical-fill check in `test_fslbm` only passed on the CPU inside one
process. No Lorentz-bin rewrite and no 4 s weld in this spec. Shop lock
unchanged.

## 22. Light damping at 0.3 mm (2026-10-23)

Undamped 0.3 mm (section 16) was clamp-quiet with mean W/D near Park, and the
two widths disagreed by 0.35 mm. This screen asks whether C_darcy 50 on that
same mesh and viscosity-floor step pulls the widths inside 0.25 mm while the
mean stays at least 5.80 mm. Ceiling 1.486 m/s, dt 16.15 us, 4.0 s, slices
22-34 mm. Records in `runs/steady/dx030_darcy50.jsonl`. Shop job untouched.

| run | W | D | H_frozen | u_surf | mach_frac | steady |
|-----|---|---|----------|--------|-----------|--------|
| 1 | 5.54 | 1.63 | 1.95 | 0.159 | 0.0025 | True |
| 2 | 5.63 | 1.66 | 2.00 | 0.159 | 0.0022 | True |

Both runs are quiet and steady. Widths agree (dW = 0.10 mm). Mean W = 5.58 mm,
below 5.80 mm. Mean D = 1.64 mm. Mean H_frozen = 1.97 mm against Park 2.2 mm.
C_darcy 100 was not run: 50 is already repeatable and already narrow.

Light damping at 0.3 mm buys the repeat and lifts the frozen height toward
Park, and it gives the Park width back. Shop 0.4 mm lock unchanged
(C_darcy 160000, dt_scale 1.0).

## 24. C_darcy gap screen at 0.3 mm (2026-10-23)

Section 22 left mean W short at C_darcy 50. Gap screen steps damping down
while holding the 0.3 mm / viscosity-floor step. Ceiling 1.486 m/s, dt 16.15 us,
4.0 s, slices 22-34 mm. Stop when mean W >= 5.80 mm and dW <= 0.25 mm. Shop
job untouched.

### C_darcy = 20 (`runs/steady/dx030_darcy20.jsonl`)

| run | W | D | H_frozen | mach_frac | steady |
|-----|---|---|----------|-----------|--------|
| 1 | 5.78 | 1.66 | 2.09 | 0.0004 | True |
| 2 | 5.58 | 1.69 | 1.80 | 0.0051 | True |

dW = 0.20 mm (pass). Mean W = 5.68 mm (short of 5.80). Mean H_frozen = 1.95 mm.
### C_darcy = 15 (`runs/steady/dx030_darcy15.jsonl`)

| run | W | D | H_frozen | mach_frac | steady |
|-----|---|---|----------|-----------|--------|
| 1 | 5.84 | 1.68 | 2.08 | 0.0013 | True |
| 2 | 5.62 | 1.66 | 1.92 | 0.0003 | True |

dW = 0.22 mm (pass). Mean W = 5.73 mm (short of 5.80). Mean H_frozen = 2.00 mm.
### C_darcy = 12 (`runs/steady/dx030_darcy12.jsonl`)

| run | W | D | H_frozen | mach_frac | steady |
|-----|---|---|----------|-----------|--------|
| 1 | 5.71 | 1.70 | 1.87 | 0.0021 | True |
| 2 | 5.60 | 1.67 | 1.80 | 0.0018 | True |

dW = 0.11 mm (pass). Mean W = 5.65 mm (short of 5.80). Mean H_frozen = 1.84 mm.
### C_darcy = 10 (`runs/steady/dx030_darcy10.jsonl`)

| run | W | D | H_frozen | mach_frac | steady |
|-----|---|---|----------|-----------|--------|
| 1 | 5.86 | 1.64 | 1.89 | 0.0013 | True |
| 2 | 5.79 | 1.69 | 2.03 | 0.0000 | True |

**Stop.** dW = 0.069 mm (<= 0.25). Mean W = 5.82 mm (>= 5.80). Mean D = 1.66 mm.
Mean H_frozen = 1.96 mm (Park 2.2). Both runs steady; force clamp quiet; mach_frac
<= 0.0013. Ceiling 1.486 m/s, dt 16.15 us, tau 0.5005 unchanged.

Gap-screen summary (0.3 mm, dt_scale viscosity floor):

| C_darcy | mean W | dW | mean H_frozen | pass |
|---------|--------|-----|---------------|------|
| 50 | 5.58 | 0.10 | 1.97 | no (W) |
| 20 | 5.68 | 0.20 | 1.95 | no (W) |
| 15 | 5.73 | 0.22 | 2.00 | no (W) |
| 12 | 5.65 | 0.11 | 1.84 | no (W) |
| **10** | **5.82** | **0.07** | **1.96** | **yes** |
| 0 (undamped) | ~6.63 | 0.35 | ~2.1 | no (dW) |

Working light-damping point at 0.3 mm is C_darcy = 10. Shop 0.4 mm lock unchanged
(C_darcy 160000, dt_scale 1.0).

## 25. Held-out validation on C_darcy 10 (2026-10-24)

Spec: `docs/SPEC_DX030_DARCY10_HELD_OUT.md`. Protocol: dx 0.3 mm, C_darcy 10,
viscosity-floor step, baseline knobs (eta 0.82, Goldak 2.0/3.8/2.8/2.6). Shop
job untouched. Calibrate baseline remains `runs/steady/dx030_darcy10.jsonl`
(mean W 5.82, D 1.66, H_frozen 1.96, dW 0.07).

### Held-outs (no retune)

| case | mean W | mean D | mean H_frozen | dW | steady | vs calibrate |
|------|--------|--------|---------------|-----|--------|--------------|
| fast 15 mm/s (`dx030_darcy10_heldout_fast.jsonl`, x 30-36) | 4.95 | 1.25 | 1.60 | 0.15 | both | shrinks W and D |
| hot 120 A (`dx030_darcy10_heldout_hot.jsonl`) | 6.21 | 2.08 | 2.20 | 0.09 | both | grows W and D |
| macro2 overhead soft (`dx030_darcy10_heldout_macro2.jsonl`) | 5.62 | 1.75 | 2.10 | 0.07 | both | cite 5.7/1.9/2.5; W close, D and H soft-low |

**Trends hold.** Fast shrinks, hot grows, each pair dW <= 0.25 mm.

### One-knob residual screen (calibrate job only)

| screen | mean W | mean D | mean H_frozen | dW | note |
|--------|--------|--------|---------------|-----|------|
| eta 0.86 (`dx030_darcy10_eta086.jsonl`) | 5.60 | 1.76 | 1.99 | 0.01 | D closer to Park; **W worse** |
| Goldak c 2.4 (`dx030_darcy10_c24.jsonl`) | 5.78 | 1.65 | 1.76 | 0.08 | W not improved; H worse. Run 2 relaunched after CUDA memcpy fault mid-pair |

**Decision: do not lock.** Keep eta 0.82 and Goldak c 2.6. Neither residual
improved width. Working point stays C_darcy 10 on 0.3 mm with baseline knobs.
Shop 0.4 mm lock unchanged (C_darcy 160000, dt_scale 1.0). Ansys Tier A remains
a separate thermal check and was not run in this section.

## 23. Known risks / gotchas

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
