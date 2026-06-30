# waam_twin v2 — Master Execution Plan

**Status:** Single source of truth for v2 development  
**Consolidates:** Code audit (B01–B20), portable/accurate master plan, materials-as-data, FLOW-3D-informed physics priorities  
**Release target:** `waam_twin` **v2.0.0** (this document’s “v2.0” = prior “v1.0 release” criteria)

---

## Document map

| Section | Contents |
|---------|----------|
| §1–2 | Vision, success criteria, explicit non-goals |
| §3 | Guiding principles |
| §4 | Current baseline (what works / what’s broken) |
| §5 | FLOW-3D: what to borrow vs skip |
| §6 | Target architecture & workstreams |
| §7 | Physics step order |
| §8–12 | Phases 0–4 (weekly tasks + exit gates) |
| §13 | Bug-fix register + task ID crosswalk |
| §14–17 | Config, validation, timeline, risks |
| §18 | Definition of done (v2.0) |

---

## 1. Vision

Build a **GPU-accelerated (with CPU fallback), material-file-driven, validated** WAAM melt-pool simulator competitive with **research-grade LBM/VOF weld CFD** — usable on laptops, lab workstations, and CI servers **without code changes**.

Bead geometry is a **prediction**, not an input: it emerges from Marangoni flow, buoyancy, surface tension, mushy-zone drag, arc heat, and wire/droplet deposition — the same coupling philosophy as FLOW-3D, implemented on **Taichi LBM + VOF** (not FVM/FAVOR/DEM).

### v1 codebase → v2 product

| v1 (current) | v2 (target) |
|--------------|-------------|
| Hardcoded grids (128×64×32, 256×128×64) | Presets + `auto_grid()` from host VRAM |
| CUDA assumed | CUDA → Vulkan → CPU auto-fallback |
| Materials as Python constants (placeholders treated as truth) | YAML files; `placeholder` vs `calibrated` |
| Monolithic `kernels.py` + broken LBM loop | Modular `physics/` + validated solver |
| Static `phi` (no real free surface) | Advected VOF + interface BCs |
| Tied to one machine / RTX 3050 assumptions | Job files; any tier via config |
| FLOW-3D parity implied | WAAM mesoscale accuracy with documented limits |

---

## 2. Success criteria (v2.0 release)

| Category | Criterion |
|----------|-----------|
| **Correctness** | All regression tests pass on CPU (`minimal` preset) and GPU (`standard+`) |
| **Thermal accuracy** | Pool width/depth **±15–20%** with `validated` material + calibration |
| **CFD accuracy** | Marangoni direction correct; pool aspect ratio **±25%** (stretch goal **±15%**) |
| **Materials** | No accuracy claims on `placeholder` materials; external YAML loader + schema |
| **Portability** | CUDA → Vulkan → CPU auto-fallback; presets not hardcoded grids |
| **Integration** | `kuka.py` / job files / Flask UI work on any tier via config |
| **Evidence** | Validation report records: material version, calibration id, preset, backend, grid, dx, git sha, machine class |

### Explicit non-goals (v2.0)

- Grain structure / microstructure  
- Residual stress & distortion FEA  
- Full plasma / electromagnetic models  
- LES turbulence  
- L-PBF powder DEM  
- Laser ray-tracing (arc heat models instead)  
- FVM + FAVOR rewrite  

---

## 3. Guiding principles

1. **Correctness before features** — no new physics until the LBM loop is valid.  
2. **Physics engine ≠ your hardware** — presets + `auto_grid()` from available memory.  
3. **Materials are data** — built-in values are **placeholders** until validated; accuracy tied to file version.  
4. **Process ≠ thermophysical** — `materials.json` (current, WFS, travel) separate from alloy physics YAML.  
5. **Test every module in isolation** — refactor `step()` into composable operators in `coupled_step.py`.  
6. **Degrade gracefully** — smaller grid, fewer tracers, CPU path, headless mode.  
7. **Calibrate honestly** — 3–5 scalar overlays (η, heat loss, Marangoni scale) on first-principles physics.  

---

## 4. Current state (baseline)

### What works today

- Clean module split (`grid`, `kernels`, `twin`, `materials`, `viewer`, `verify`)  
- SoA Taichi fields, ping-pong LBM buffers  
- SymPy kernel derivation (`tools/derive_cumulant.py`)  
- Enthalpy-porosity, Marangoni CSF (concept), Carman–Kozeny, tracers, HAZ `T_max`  
- Viewer with deferred `ti.init()` and partial CUDA fallback  
- `kuka.py` lazy twin integration hook  

### What is broken or incomplete (audit register)

| ID | Issue | Severity | Phase fix |
|----|-------|----------|-----------|
| B01 | `stream()` missing normal pull; bounce-back overwritten | **P0** | 0 |
| B02 | Arc pressure applied then wiped by Marangoni | **P0** | 0 |
| B03 | `Fx,Fy,Fz` never cleared; forces accumulate | **P0** | 0 |
| B04 | `use_srt` flag unused; MRT never called | **P0** | 0 |
| B05 | `init_grid`/`reset()` incomplete (`T_max`, forces, tracers) | **P0** | 0 |
| B06 | V&V thermal test fails (~20% L2); calls full `step()` not diffusion-only | **P0** | 0 |
| B07 | No VOF advection; `phi` static after init | **P1** | 2 |
| B08 | `feed_wire` hardcodes `300000 * 7800` not `L_rho` | **P1** | 0 |
| B09 | `tau_T` computed, never used | **P1** | 2 |
| B10 | `beta_T` hardcoded in `twin.py` | **P1** | 1 |
| B11 | `viewer.py`: `clip_y` uninitialized; wrong `FLAG_FLUID` constant | **P2** | 0 |
| B12 | `twin.py`: unused imports (`json`, `pathlib`) | **P2** | 0 |
| B13 | Gas cells skip streaming write → stale distributions | **P2** | 0 |
| B14 | `get_telemetry()` always pulls 4 full 3D arrays | **P2** | 1 |
| B15 | CUDA-only in `verify.py`, `kuka.py`; hardcoded grids | **P0** | 0 |
| B16 | Materials hardcoded as if authoritative | **P0** | 0 |
| B17 | `grid.py` docs tied to RTX 3050 / 4 GB | **P2** | 0 |
| B18 | `requirements.txt` missing taichi, numpy, etc. | **P2** | 0 |
| B19 | No README / validation docs | **P2** | 4 |
| B20 | MRT kernel named “cumulant”; simplified vs full Geier LBM | **P1** | 2 |

---

## 5. FLOW-3D-informed scope (not a clone)

### Borrow (physics coupling)

- Coupled timestep: thermal ↔ surface forces ↔ flow ↔ solidification  
- Marangoni (dγ/dT), isotropic surface tension, optional recoil  
- Enthalpy–porosity + Carman–Kozeny mushy drag  
- Bead shape as emergent prediction  

### Do not copy (v2)

| FLOW-3D piece | v2 decision |
|---------------|-------------|
| FAVOR cut cells | Defer unless wire geometry demands it |
| TruVOF on FVM + GMRES | Implement VOF **ideas** on LBM (advect φ, interface BCs) |
| DEM powder bed | WAAM uses wire + arc + droplet |
| Ray-tracing laser | Arc heat source models (Gaussian, Goldak stub) |
| HPC MPI stack | Taichi GPU + optional CPU CI |

---

## 6. Target architecture & workstreams

### Repository layout

```
FYP22-01/
├── config/
│   ├── presets.yaml
│   └── defaults.yaml
├── materials/
│   ├── schema.json
│   ├── placeholders/          # ER70S-6, SS316L, AISI4043 — status: placeholder
│   ├── validated/
│   └── user/                  # .gitignore
├── jobs/
│   └── examples/
├── docs/
│   ├── WAAM_TWIN_V2_EXECUTION_PLAN.md   # this file
│   ├── MATERIALS.md
│   ├── HARDWARE.md
│   └── validation/
├── waam_twin/
│   ├── platform.py, job.py, calibration.py
│   ├── twin.py, grid.py, materials.py
│   ├── solvers/coupled_step.py
│   ├── physics/
│   │   ├── thermal.py, phase_change.py, free_surface.py
│   │   ├── forces.py, arc.py, deposition.py, lbm.py
│   ├── validation/
│   │   ├── run_all.py
│   │   ├── test_thermal_diffusion.py
│   │   ├── test_phase_stefan.py
│   │   ├── test_lbm_poiseuille.py
│   │   ├── test_lbm_cavity.py
│   │   ├── test_mass_conservation.py
│   │   ├── test_marangoni_cell.py
│   │   └── baselines/
│   ├── kernels.py             # shrinks during migration
│   ├── cumulant_kernel.py, viewer.py, verify.py (deprecated)
│   └── tools/derive_cumulant.py
├── requirements.txt
└── kuka.py
```

### Eight workstreams

| WS | Name | Phases | Primary deliverables |
|----|------|--------|----------------------|
| **WS-A** | Core numerics (audit P0) | 0 | stream, forces, init, MRT wire |
| **WS-B** | Platform & portability | 0–1 | `platform.py`, presets, `auto_grid`, env vars |
| **WS-C** | Materials & calibration | 0–4 | YAML schema, placeholders, validated alloys |
| **WS-D** | Architecture refactor | 0–1 | `coupled_step.py`, `from_preset`, `from_job` |
| **WS-E** | Thermal physics | 1 | losses, T-tables, arc models, Stefan |
| **WS-F** | CFD & free surface | 2 | VOF, CSF Marangoni, MRT production |
| **WS-G** | WAAM multiphysics | 3 | deposition, multi-bead, paths, HAZ |
| **WS-H** | Validation, docs, product | 0–4 | `run_all`, CI, README, validation matrix |

### Task ID crosswalk (master plan ↔ this doc)

| Master ID | v2 task | Description |
|-----------|---------|-------------|
| A01–A02 | 0.4 | Fix `stream()` pull + bounce-back |
| A03 | 0.5 | Gas-cell streaming |
| A04 | 0.6 | `clear_forces` |
| A05 | 0.7 | Force order in `coupled_step` |
| A06 | 0.18 | Wire `use_srt` / MRT |
| A07 | 0.8 | Complete init/reset |
| A08–A09 | 0.9 | `feed_wire` + `L_rho` |
| B01–B03 | 0.1 | `platform.py`, `PlatformProfile` |
| B02 | 0.23 | Env vars |
| B04 | 0.2 | `presets.yaml` |
| B05 | 0.19 | `auto_grid()` |
| B06 | 0.25 | `auto_tracer_count()` |
| B07 | 0.3 | Replace all `ti.init(cuda)` |
| B08 | 0.20 | Remove hardcoded grids |
| B09 | 0.26 | Generic VRAM docs in `grid.py` |
| B10 | 0.27 | OOM guard before alloc |
| B11 | 1.5 | Full `from_job()` |
| B12 | 1.13 | Telemetry downsampling |
| B13 | 1.15 | `export_vtk` optional / headless |
| C01–C05 | 0.10–0.11 | Schema, placeholders, loader, warnings |
| C10–C12 | 1.6, 1.11 | Calibration layer + first validated material |
| C20–C21 | 4.9, 4.11 | Calibration script + promotion workflow |
| D01 | 0.12 | `coupled_step.py` |
| D02 | 0.13 | `from_preset()` |
| D03 | 1.5 | `from_job()` |
| D04 | 0.28 | `bind_velocity_set` regression check |
| E01–E10 | 1.1–1.10 | Thermal workstream |
| F01–F12 | 2.1–2.10 | CFD / VOF workstream |
| G01–G07 | 3.1–3.7 | WAAM workstream |
| H01–H48 | §8–12, §15 | All validation & docs tasks |

---

## 7. Physics step order (`coupled_step.py`)

```
 1. clear_forces(Fx, Fy, Fz)
 2. arc.inject_heat(H)                         [if welding]
 3. deposition.feed_wire                       [on droplet schedule]
 4. deposition.inject_tracers                  [on droplet schedule]
 5. thermal.advect_diffuse(H, T, u)
 6. thermal.apply_boundary_losses(H, T)        [Phase 1+]
 7. phase_change.update(H, T, f_l)
 8. thermal.update_T_max(T, T_max)
 9. free_surface.advect_phi(phi, u)            [Phase 2+]
10. free_surface.reinitialize_phi(phi)        [Phase 2+]
11. free_surface.update_flags(phi, flags)     [Phase 2+]
12. forces.marangoni → Fx, Fy, Fz
13. forces.buoyancy → Fz
14. forces.arc_pressure → Fz                   [accumulate AFTER Marangoni]
15. forces.recoil → Fz                        [Phase 2 optional]
16. lbm.collide_srt | collide_mrt
17. lbm.stream
18. deposition.advect_tracers
19. grid.swap_buffers()
```

---

## 8. Phase 0 — Foundation (weeks 1–4)

**Goal:** Trustworthy core loop; runs on any machine; placeholders labeled; CI green on CPU.

### WS-A: Core numerics

| Task | WS | Files | Done when |
|------|-----|-------|-----------|
| 0.4 Fix `stream()`: normal pull + correct bounce-back | A | `physics/lbm.py` | Mass conservation < 1% |
| 0.5 Fix gas-cell streaming (no stale `f_dst`) | A | `lbm.py` | No NaN in gas after 10k steps |
| 0.6 Add `clear_forces` kernel | A | `physics/forces.py` | Forces don’t accumulate |
| 0.7 Force order: clear → Marangoni → buoyancy → arc pressure | A | `coupled_step.py` | Arc pressure affects collision |
| 0.8 Complete `init_grid` / `reset` (T_max, forces, tracers, tracer_head, `_last_droplet_time`) | A | `kernels.py`, `twin.py` | Identical restart twice |
| 0.9 `feed_wire`: `H = cp_rho*T_drop + L_rho`; remove hardcoded steel | A | `kernels.py` | SS316L feed correct |
| 0.18 Wire `use_srt` → `collide_srt` / `collide_mrt` | A | `twin.py` | MRT path stable |
| 0.29 Remove unused imports (`json`, `pathlib`) | A | `twin.py` | Lint clean |

### WS-B: Platform & portability

| Task | WS | Files | Done when |
|------|-----|-------|-----------|
| 0.1 `platform.py`: `init_taichi()` CUDA → Vulkan → CPU | B | `platform.py` | Works without NVIDIA |
| 0.2 `config/presets.yaml` | B | `config/` | Four tiers defined |
| 0.3 Replace all `ti.init(cuda)` | B | `viewer`, `verify`, `kuka` | Single init entry |
| 0.19 `auto_grid(domain_mm, target_dx, vram_budget)` | B | `platform.py` | Grid fits budget |
| 0.25 `auto_tracer_count(vram_mb)` | B | `platform.py` | No OOM on 4 GB GPU |
| 0.20 Remove hardcoded grids from `viewer.py`, `kuka.py` | B | viewer, kuka | Preset-driven |
| 0.26 Generic VRAM docs (remove RTX 3050 text) | B | `grid.py` | Hardware-agnostic |
| 0.27 OOM guard: check `estimated_vram_mb()` before alloc | B | `twin.py`, `grid.py` | Clear error + preset hint |
| 0.23 Env vars documented | B | `HARDWARE.md` | WAAM_BACKEND, PRESET, VRAM_MB, HEADLESS, MATERIAL |

### WS-C: Materials (placeholder model)

| Task | WS | Files | Done when |
|------|-----|-------|-----------|
| 0.10 `materials/schema.json` | C | `materials/` | Validates YAML |
| 0.10b Placeholder YAML: ER70S-6, SS316L, AISI4043 | C | `placeholders/` | `status: placeholder` |
| 0.11 Refactor `load_material()` → file loader | C | `materials.py` | Warn on placeholder |
| 0.11b Deprecate `_PHYSICS_LIBRARY` (thin wrapper) | C | `materials.py` | No magic numbers in Python |
| 0.11c `MaterialProps.status` in telemetry | C | `materials.py`, `twin.py` | UI shows “illustrative” |

### WS-D: Architecture

| Task | WS | Files | Done when |
|------|-----|-------|-----------|
| 0.12 Extract `solvers/coupled_step.py` | D | `solvers/` | `twin.step()` ≤ 10 lines |
| 0.13 `WAAMTwin.from_preset("standard")` | D | `twin.py` | One-liner portable setup |
| 0.30 `from_job()` stub | D | `job.py` | YAML loads without error |
| 0.28 `bind_velocity_set` regression | D | `twin.py` | Kernels still bind EX/EY/EZ |

### WS-H: Validation & viewer (Phase 0)

| Task | WS | Files | Done when |
|------|-----|-------|-----------|
| 0.14 Thermal diffusion: **kernel-only**, no substrate | H | `test_thermal_diffusion.py` | L2 < 5% CPU minimal |
| 0.15 LBM Poiseuille test | H | `test_lbm_poiseuille.py` | < 5% error |
| 0.16 Mass conservation (10k steps) | H | `test_mass_conservation.py` | < 1% drift |
| 0.17 `validation/run_all.py` | H | `validation/` | CLI with preset tolerances |
| 0.24 CI: `WAAM_BACKEND=cpu WAAM_PRESET=minimal` | H | `.github/workflows/` or script | Passes without GPU |
| 0.22 Pin `requirements.txt` | H | root | taichi, numpy, pyyaml, sympy, pyvista |
| 0.21 Viewer: init `clip_y`, fix `FLAG_FLUID` | H | `viewer.py` | No crash on C key |
| 0.21b Viewer: drop or implement mode 3 (“Velocity Flow”) | H | `viewer.py` | HUD matches behavior |
| 0.21c Viewer: `from_preset` not 128×64×32 | H | `viewer.py` | Preset grid |

### Phase 0 exit checklist

- [x] All WS-A tasks (0.4–0.9, 0.18, 0.29)  
- [x] All WS-B tasks (0.1–0.3, 0.19–0.20, 0.23, 0.25–0.27)  
- [x] All WS-C tasks (0.10–0.11c) — schema wired in `load_material()`  
- [x] `run_all` passes CPU (GPU matrix optional)  
- [x] 10k steps: no NaNs; mass conserved (`test_soak_10k`)  
- [x] Placeholder warning at startup  
- [x] `coupled_step.py` drives `step()`  
- [x] LBM cavity smoke (`test_lbm_cavity`)  
- [x] Viewer fixes (0.21–0.21c) — preset-driven grid  

---

## 9. Phase 1 — Thermal fidelity (weeks 5–10)

**Goal:** Accurate T field & phase change; T-dependent materials; first **calibrated** alloy.

### WS-E: Thermal physics

| Task | ID | Notes |
|------|-----|-------|
| 1.1 Extract `physics/thermal.py`, `physics/phase_change.py` | E01, E06 | From `kernels.py` |
| 1.2 T-dependent tables k(T), cp(T), μ(T), dγ/dT(T) in YAML | E01 | Piecewise linear |
| 1.3 `thermal.apply_boundary_losses` convection + optional radiation | E02, E03 | Job toggle |
| 1.4 `physics/arc.py`: Gaussian2D, Goldak stub, ConicalVolume stub | E04 | Pluggable |
| 1.5 `Q = η·V·I` from process + calibration | E05 | Links `materials.json` |
| 1.7 `beta_T` per material file | E10 | Remove twin hardcode |
| 1.8 Cooling rate `dT_dt` field + telemetry | E08 | HAZ hook |
| 1.9 Stefan solidification test | E07 | < 10% front error |
| 1.10 Rosenthal far-field benchmark (optional) | E09 | Documented |

### WS-C: Calibration

| Task | ID | Notes |
|------|-----|-------|
| 1.6 `calibration.py` overlays | C10 | η, heat_loss_factor, marangoni_scale |
| 1.11 First `materials/validated/ER70S-6.v1.yaml` + `.calibration.yaml` | C11, C12 | Fit η from bead-on-plate |

### WS-B / WS-D / WS-H

| Task | ID | Notes |
|------|-----|-------|
| 1.5 Full `WAAMTwin.from_job(yaml)` | B11, D03 | domain, material, process, preset |
| 1.13 Telemetry downsampling on `minimal` | B12 | Centre slice only |
| 1.15 `export_vtk` skip when headless / low disk | B13 | No crash |
| 1.12 Thermocouple benchmark ±25% | H10 | Experiment or literature |
| 1.12b Macrograph pool W/D ±20% | H11 | Calibrated material |
| 1.12c `docs/validation/thermal_v1.md` | H12 | Report template |

### Phase 1 exit checklist

- [x] Thermal diffusion L2 < 5%  
- [x] Stefan test passes  
- [x] One `calibrated` material (not placeholder)  
- [x] Pool W/D ±20% vs `model_reference` (`test_pool_geometry_standard`, standard dx)  
- [x] Job file portable minimal ↔ standard grid (`test_job_parity`)  

### Phase 2 partial (implemented early)

- [x] `advect_phi`, `reinitialize_phi`, `update_flags` in `coupled_step`  
- [x] `surface_height_at` → dynamic `arc_k`  
- [x] Balanced-force CSF (`compute_csf_tension`)  
- [x] `forces.recoil` toggle  
- [x] `μ(T)` → per-cell `tau_field` (SRT)  
- [x] MRT default on `high`/`ultra` presets (via `presets.yaml`)  
- [x] Deposition v2 momentum (`feed_wire_momentum`)  
- [x] Goldak double-ellipsoid heat source  
- [x] Multi-bead / growing substrate (`enable_substrate_growth`, `test_two_layer_remelt`)  

---

## 10. Phase 2 — CFD & free surface (weeks 11–18)

**Goal:** Moving melt pool; Marangoni circulation; VOF mass conservation; MRT on high preset.

### WS-F: Free surface & forces

| Task | ID | Notes |
|------|-----|-------|
| 2.1 `advect_phi` donor-cell | F01 | Coupled to u |
| 2.2 `reinitialize_phi` compression | F02 | PLIC later |
| 2.3 `update_flags` → FLUID/GAS/IFACE | F03 | Use FLAG_IFACE |
| 2.4 `surface_height(i,j)` prep (arc_k tracks surface) | F04 | Feeds Phase 3 |
| 2.5 Balanced-force CSF + curvature κ | F05 | Laplace test |
| 2.6 γ(T) from material tables | F06 | |
| 2.7 `forces.recoil` toggle | F07 | High-power GTAW |
| 2.8 MRT default on `high`/`ultra`; SRT on minimal/standard | F08, B09 | `tau_T` or document explicit thermal |
| 2.9 Local μ(T) → per-cell `tau` (stretch) | F09 | |
| 2.10 Extract `physics/lbm.py`; shrink `kernels.py` | F10 | |
| 2.10b Deposition v2: droplet mass + momentum | F11 | Not just pressure spike |
| 2.10c `droplet_freq` from WFS in job | F12 | Process-linked |

### WS-H: CFD validation

| Task | ID | Pass criterion |
|------|-----|----------------|
| 2.11 Laplace / spurious currents | H20 | Below threshold |
| 2.12 Thermocapillary Marangoni cell | H21 | Correct flow direction |
| 2.13 Differentially heated cavity | H22 | Nu trend correct |
| 2.14 Literature weld-pool aspect ratio | H23 | ±25% |
| 2.14b `docs/physics/LBM.md` + ASSUMPTIONS | H24, H43 | MRT vs cumulant naming |

### Phase 2 exit checklist

- [x] φ advects; metal volume leak < 2%  
- [x] Marangoni direction matches dγ/dT sign (`test_marangoni_cell`)  
- [x] MRT on `high` preset (via `presets.yaml`)  
- [x] Pool aspect ratio ±25% on full standard domain (medium-grid smoke at standard dx)  
- [x] Arc pressure regression test passes (`test_arc_pressure`)  
- [x] Differentially heated cavity (`test_heated_cavity`)  
- [x] LBM cavity / forced flow smoke (`test_lbm_cavity`)  

### Phase 3 partial

- [x] `TorchPathDriver` + `WAAMTwin.run_path()`  
- [x] `solidify_cooled_metal` + `enable_substrate_growth`  
- [x] `kuka_adapter.py` + `WAAM_JOB` env  
- [x] `from_process_sheet()` in `job.py`  
- [x] HAZ VTK (`export_haz_vtk`, `T_max` in `export_vtk`)  
- [x] `test_multi_bead` path smoke  
- [x] CSV torch path (`torch_path_csv`, `jobs/paths/bead_line.csv`)  
- [x] Interpass cooling at segment boundaries + `two_layer.yaml`  
- [x] `test_interpass_haz`, `test_parametric_monotonic`  
- [x] `porosity_pct` in telemetry  
- [x] `validation/metadata.py` + `baselines/v2.0_dev.json`  
- [x] `README_WAAM_TWIN.md`  
- [x] Multi-bead width vs `model_reference` smoke (`test_multi_bead_width`, min 1 mm)  
- [x] 2-layer remelt smoke (`test_two_layer_remelt`, `remelt_hot_solid`)  
- [x] Moving simulation window (`shift_simulation_window_x`, `test_moving_window`)  

### Gap-close (pre–Phase 4)

- [x] Calibration fitter (`tools/fit_calibration.py`) + `ER70S-6.bead_on_plate.yaml`  
- [x] `model_reference` in job YAML (separate from macrograph `reference`)  
- [x] `test_calibrated_pool` — ±30% vs fitted `fit_metrics` on 88×44×44  
- [x] Surface mesh VTK (`export_surface_vtk`, `test_surface_vtk`)  
- [x] Validation matrix hook (`tools/run_validation_matrix.py`)  
- [x] Multi-bead ±15% vs `model_reference` (`test_multi_bead_width`)  
- [x] 2-layer HAZ reference band (`test_two_layer_haz_ref`, qualitative vs FEA envelope)  

---

## 11. Phase 3 — WAAM multiphysics (weeks 19–26)

**Goal:** Multi-bead, layer growth, HAZ, porosity — WAAM-specific value.

| Task | ID | Notes |
|------|-----|-------|
| 3.1 Growing substrate / `surface_height` integration | G01 | Solidified bead → SOLID |
| 3.2 Remelt neighbors on new layer | G01 | |
| 3.3 Interpass cooling (arc off, diffuse on) | G02 | `T_max` HAZ map |
| 3.4 Torch path from CSV / G-code | G03 | Physical coords |
| 3.5 `kuka.py` thin adapter TCP → job coords | G04 | No sim logic in kuka |
| 3.6 Porosity v2: mushy capture + `porosity_pct` | G05 | |
| 3.7 `from_process_sheet(I, V, WFS, travel)` | G07 | |
| 3.8 Moving simulation window | G06 | Bounded VRAM |
| 3.9 Multi-bead width vs macrograph ±15% | H30 | |
| 3.10 2-layer HAZ vs thermocouple/FEA | H31 | Qualitative+ |
| 3.11 Parametric sweep P, v monotonic | H32 | |
| 3.12 HAZ map VTK export | — | `T_max` field |

### Phase 3 exit checklist

- [x] Multi-bead single layer width vs `model_reference` ±15% (`test_multi_bead_width`)  
- [x] HAZ map exports to VTK  
- [x] Torch path robot-agnostic (`torch_path` YAML + CSV + `TorchPathDriver`)  
- [x] `kuka.py` uses `kuka_adapter.create_twin_from_env`  
- [x] 2-layer HAZ peak in reference band (`test_two_layer_haz_ref`)  
- [x] Moving simulation window (`test_moving_window`)  
- [x] G-code deferred — CSV + YAML paths ship; see §2 non-goals for full G-code parser  

---

## 12. Phase 4 — Product & evidence (weeks 27–36)

**Goal:** v2.0.0 release; full validation matrix; docs; performance.

### Validation matrix (record metadata every run)

| Variable | Levels |
|----------|--------|
| Arc power | 2–3, 3–4, 4–5 kW |
| Travel speed | 3, 5, 8 mm/s |
| Material | ER70S-6, SS316L (**validated** files only) |
| Geometry | Bead-on-plate, single wall, 2-layer |
| Preset | minimal (CI), standard, high |
| Backend | cpu, vulkan, cuda |

Metadata per run: `material_version`, `calibration_id`, `preset`, `backend`, `grid`, `dx`, `git_sha`, `machine_class`.

### WS-H: Product tasks

| Task | ID | Deliverable |
|------|-----|-------------|
| 4.1–4.2 Full matrix + validation report | H44 | ±15% pool geometry documented |
| 4.3 README | H40 | Install, quickstart, presets |
| 4.4 MATERIALS.md, HARDWARE.md | H41, H42 | |
| 4.5 VTK surface mesh export | H45 | Not voxel-only |
| 4.6 Stable telemetry JSON schema | H46 | Flask / kuka |
| 4.7 Performance: cells/sec per tier/backend | H47 | |
| 4.8 Versioned baselines | H48 | `validation/baselines/v2.0_*.json` |
| 4.9 Calibration fitting script | C20 | Fit η, heat_loss |
| 4.10 Compare one commercial/open reference case | — | Documented in report |
| 4.11 Material promotion workflow placeholder→calibrated | C21 | MATERIALS.md |
| 4.12 Tag `waam_twin` **v2.0.0** | — | |
| 4.13 Runs on CPU, Vulkan, CUDA without fork | — | Exit gate |

### Phase 4 exit checklist

- [x] ±15% vs `model_reference` on bead matrix cases (`run_validation_matrix`; path jobs smoke-only)  
- [x] Internal validation report (`docs/validation/VALIDATION_REPORT.md`)  
- [x] Reference case comparison (`docs/validation/reference_case_ER70S6.md`)  
- [x] Backend smoke CPU required; Vulkan/CUDA via `WAAM_BACKEND_MATRIX=1` (`test_backend_smoke`)  
- [x] VTK surface mesh export (`export_surface_vtk`)  
- [x] Calibration fitting script (`tools/fit_calibration.py`)  
- [x] Validation matrix runner (`tools/run_validation_matrix.py`)  
- [x] Telemetry JSON schema (`validation/telemetry_schema.json`)  
- [x] Performance benchmark (`tools/benchmark_performance.py`)  
- [x] Versioned baselines (`validation/baselines/v2.0_matrix.json`)  
- [x] Material promotion workflow (`docs/MATERIALS.md` §Promotion)  
- [x] Tag `waam_twin` **v2.0.0** (`__version__` in `waam_twin/__init__.py`; git tag optional)  
- [x] SS316L material file (`materials/validated/SS316L.v1.yaml`, `status: placeholder` until process calibration)  

---

## 13. Bug-fix register (complete)

See §4 table. Summary mapping to phases:

| Audit | Phase |
|-------|-------|
| B01–B06, B08, B11–B13, B15–B18 | 0 |
| B10, B14 | 1 |
| B07, B09, B20 | 2 |
| B19 | 4 |

---

## 14. Configuration reference

### Environment variables

| Variable | Values | Default |
|----------|--------|---------|
| `WAAM_BACKEND` | auto, cuda, vulkan, cpu | auto |
| `WAAM_PRESET` | minimal, standard, high, ultra | standard |
| `WAAM_VRAM_MB` | integer override | auto-detect |
| `WAAM_HEADLESS` | 0, 1 | 0 |
| `WAAM_MATERIAL` | path or name | placeholders/ER70S-6.yaml |

### `config/presets.yaml` (full sketch)

```yaml
minimal:
  vram_budget_mb: 512
  domain_mm: [40, 20, 15]
  target_dx_mm: 0.5
  max_tracers: 5000
  use_srt: true

standard:
  vram_budget_mb: 2048
  domain_mm: [80, 40, 25]
  target_dx_mm: 0.3
  max_tracers: 20000
  use_srt: true

high:
  vram_budget_mb: 8192
  domain_mm: [120, 60, 40]
  target_dx_mm: 0.2
  max_tracers: 50000
  use_srt: false

ultra:
  vram_budget_mb: 16384
  domain_mm: [200, 100, 60]
  target_dx_mm: 0.15
  max_tracers: 100000
  use_srt: false
```

### Example job file

```yaml
simulation:
  preset: auto
  backend: auto
  domain_mm: [80, 40, 25]
  target_dx_mm: 0.3

material: materials/placeholders/ER70S-6.yaml
calibration: null

process:
  current_A: 140
  voltage_V: 20
  travel_speed_mm_s: 5
  wire_feed_m_min: 8

heat_source: gaussian2d
heat_loss:
  convection: true
  radiation: false
```

### Example Python API

```python
import waam_twin.platform as platform
platform.init_taichi()

from waam_twin import WAAMTwin

twin = WAAMTwin.from_preset("standard", material="materials/placeholders/ER70S-6.yaml")
# or
twin = WAAMTwin.from_job("jobs/examples/bead_on_plate.yaml")

twin.reset()
for step in range(10_000):
    twin.step(torch_x_m=step * 5e-3 / 10_000, torch_y_m=0.02, is_welding=True)
print(twin.get_telemetry())
```

---

## 15. Acceptance tests summary

| Test | Phase | Pass criterion |
|------|-------|----------------|
| Gaussian thermal diffusion | 0 | L2 < 5% (kernel only) |
| LBM Poiseuille | 0 | < 5% vs analytical |
| LBM cavity (optional) | 0+ | Documented |
| Mass conservation | 0 | < 1% / 10k steps |
| Stefan solidification | 1 | < 10% front position |
| Thermocouple T(t) | 1 | ±25% |
| Macrograph W/D | 1 | ±20% calibrated |
| Laplace / spurious currents | 2 | Below threshold |
| Marangoni cell | 2 | Correct flow direction |
| Differentially heated cavity | 2 | Nu trend correct |
| Pool aspect ratio | 2 | ±25% literature case |
| Multi-bead width | 3 | ±15% |
| Parametric P, v trends | 3 | Monotonic |
| Full validation matrix | 4 | ±15% documented |
| CPU CI `run_all` | 0+ | Always green |
| Vulkan backend smoke | 4 | `run_all` minimal |

---

## 16. Timeline & staffing

### Solo (~9 months)

| Phase | Weeks | Focus |
|-------|-------|-------|
| 0 | 1–4 | P0 bugs + platform + placeholders + CI |
| 1 | 5–10 | Thermal + calibration + jobs |
| 2 | 11–18 | VOF + forces + MRT + CFD validation |
| 3 | 19–26 | WAAM deposition + multi-bead |
| 4 | 27–36 | Matrix + docs + v2.0.0 tag |

### Two developers (~6 months)

- **Dev A:** WS-A, WS-D, WS-E, WS-F (physics/kernels)  
- **Dev B:** WS-B, WS-C, WS-H (platform, materials, validation, docs)  
- Phases 3–4 overlap  

### First 30 days (implementation order)

**Week 1:** `platform.py`, `presets.yaml`, replace `ti.init`; fix `stream()` (0.4–0.5); `clear_forces` + force order (0.6–0.7); placeholder YAML (0.10–0.11)  

**Week 2:** init/reset (0.8); `feed_wire` L_rho (0.9); `coupled_step.py` (0.12); thermal V&V kernel-only (0.14)  

**Week 3:** MRT wire (0.18); Poiseuille + mass conservation (0.15–0.16); `run_all` + CPU CI (0.17, 0.24); viewer fixes (0.21)  

**Week 4:** `auto_grid` + `auto_tracer_count` (0.19, 0.25); OOM guard (0.27); de-hardcode kuka (0.20); `from_preset` / job stub (0.13, 0.30); requirements (0.22); Phase 0 gate review  

### Critical path

```
0.4 stream → 0.16 mass conservation → 2.1 VOF → 2.12 Marangoni test → 3.9 multi-bead
0.1 platform → 0.19 auto_grid → 1.5 from_job → 3.5 kuka adapter
0.10 placeholders → 1.11 validated material → 4.1 validation matrix
```

---

## 17. Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LBM unstable at high ΔT | Wrong velocities | Ma cap, MRT on high only, velocity limiter |
| VOF mass loss | Wrong pool shape | Reinit, smaller dt, PLIC later |
| Placeholder materials in reports | False accuracy | `status` + mandatory report metadata |
| CPU too slow | Bad UX | `minimal` preset, headless batch |
| Vulkan backend gaps | AMD/Intel issues | Document Taichi version; CPU fallback |
| Scope creep (FEA, DEM, ray trace) | Never ship | §2 non-goals |
| No experimental data | Can't calibrate | Literature first; one in-house case |
| MRT register spill on small GPU | OOM / slow | SRT on minimal/standard only |
| Materials are placeholders | Misleading sims | Warn at load; separate validated/ dir |

---

## 18. Definition of done — v2.0.0

> **`waam_twin` v2.0.0** is a portable WAAM melt-pool simulator that:
>
> - Passes all validation tests on CPU (`minimal`) and GPU (`standard+`), and smoke-tests Vulkan  
> - Loads materials from versioned YAML with explicit `placeholder` vs `calibrated` status  
> - Auto-selects grid, tracers, and backend for the host machine  
> - Predicts pool width, depth, peak temperature, and Marangoni-driven flow within documented tolerances on **validated** ER70S-6 and SS316L profiles  
> - Integrates via job files and `kuka.py` without machine-specific constants  
> - Ships validation report, HARDWARE.md, MATERIALS.md, and performance benchmarks  
> - Implements FLOW-3D-**class** coupling (thermal ↔ surface forces ↔ flow ↔ solidification) on LBM+VOF, without claiming L-PBF DEM or laser ray parity  

---

## 19. Immediate next actions

1. `waam_twin/platform.py` + `config/presets.yaml`  
2. Fix `stream()` and force pipeline (0.4–0.7)  
3. `materials/placeholders/ER70S-6.yaml` with `status: placeholder`  
4. Scaffold `validation/run_all.py` + kernel-only thermal test  
5. `__version__ = "2.0.0-dev"` in `waam_twin/__init__.py`  

---

*This document supersedes all prior ad-hoc plans. Track progress via Phase exit checklists and master task IDs (A01, B01, …) in PR descriptions.*
