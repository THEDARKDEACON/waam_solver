# WAAM Twin — Bead Geometry Physics Spec

**Status:** Draft v1.0  
**Audience:** Physics/kernel implementers, validation authors, job YAML maintainers  
**Scope:** Physics-first bead crown, wetting, deposition inlet, gravity, solidification locking, multi-layer CTWD  
**Depends on:** `coupled_step.py`, `kernels.py`, `physics/forces.py`, `physics/deposition.py`, VOF/CSF stack  

This document specifies how to replace scenario-specific calibration of bead shape with **first-principles free-surface and deposition physics**. Calibration overlays (`arc_sigma_scale`, `marangoni_scale`) remain for arc distribution uncertainty only — **not** for bead height or toe angle.

---

## 1. Goals

| Goal | Description |
|------|-------------|
| **Predictive bead geometry** | Crown height, width, toe angle, penetration emerge from process + material inputs |
| **Substrate wetting** | Contact angle θ (or surface energies) at solid–liquid–gas triple line |
| **Physical mass entry** | Wire volume enters at pool interface with correct enthalpy/momentum — no gas-column fill |
| **Hydrostatic flattening** | Full gravity on liquid free surface competes with γ |
| **Freeze locking** | Solidification captures spread before next droplet |
| **Multi-layer path** | CTWD / stick-out resistance feeds wire preheat and arc energy partition (later tier) |

### Non-goals (this spec)

- Replacing LBM/VOF with FVM/FAVOR  
- Full short-circuit GMAW arc plasma model  
- Grain microstructure, residual stress FEA  
- Closed-loop machine control (spec defines **simulated** CTWD coupling only)  
- Using `η_d` or geometry calibration scalars to match one bead-on-plate macrograph  

---

## 2. Problem statement

### 2.1 Observed symptom

Single-bead and bead-on-plate runs produce **excessive crown height** relative to macrographs: mass is conserved (`mass_balance` ≈ 1.04) but metal piles vertically instead of spreading laterally with a realistic toe angle.

### 2.2 Root causes in current code

| Gap | Location | Effect |
|-----|----------|--------|
| No wetting BC at solid walls | `compute_csf_tension` skips `FLAG_SOLID` | Liquid does not feel substrate contact angle |
| Deposition fills gas column | `feed_wire` `in_column` branch | Artificial vertical metal stack above pool |
| Buoyancy is thermal only | `add_buoyancy` (Boussinesq βΔT) | No ρg hydrostatic flattening of crest |
| Calibration targets pool W/D only | `fit_calibration.py`, `model_reference` | Crown height never gated |
| Triple line not resolved | CSF on φ only; solid top = step | Toe angle smeared at coarse dx |

### 2.3 Design principle

> **Inputs:** material γ(T), μ(T), ρ, L, k, cp from YAML; process I, V, v_wire, v_travel; **one** wetting descriptor θ_wet per material/surface condition.  
> **Outputs:** bead cross-section metrics (w, h, θ_toe, penetration) must change correctly when travel speed or wire feed changes **without refitting**.

---

## 3. Work packages overview

| ID | Package | Priority | Blocks |
|----|---------|----------|--------|
| **WP1** | Substrate wetting / contact angle CSF | P0 | Realistic toe, lateral spread |
| **WP2** | Physical deposition inlet | P0 | Remove artificial crown |
| **WP3** | Hydrostatic gravity | P1 | Crest sag on horizontal plate |
| **WP4** | Solidification–flow coupling | P1 | Lock spread; layer height stability |
| **WP5** | CTWD / stick-out resistance (multi-layer) | P2 | Layer-to-layer arc energy drift |

**Recommended implementation order:** WP1 → WP2 → WP3 → WP4 → WP5.

---

## 4. WP1 — Substrate wetting (contact angle)

### 4.1 Physics

At the solid–liquid–gas triple line, Young's law:

\[
\cos\theta = \frac{\gamma_{sv} - \gamma_{sl}}{\gamma_{lv}}
\]

For implementation, specify **`contact_angle_deg`** θ in material or job `surface_wetting` block (default: literature value for oxide-free steel-on-steel ≈ 70–90°; high-S steels may differ).

**CSF contact-angle model** (Brackbill et al., 1992; standard in VOF weld CFD):

At cells adjacent to `FLAG_SOLID`, modify the **interface normal** used in CSF so the liquid–gas interface meets the wall at angle θ:

\[
\hat{n}_\text{eff} = \sin\theta\,\hat{n}_\text{wall} + \cos\theta\,\hat{t}_\text{wall}
\]

where \(\hat{n}_\text{wall}\) is the outward solid normal (typically +ẑ on substrate top) and \(\hat{t}_\text{wall}\) lies in the tangent plane.

Capillary pressure at the wall:

\[
p_\gamma = \gamma \, \kappa_\text{eff}
\]

Force density (existing convention): \(\mathbf{F} = \gamma \kappa_\text{eff} \nabla\phi\).

### 4.2 Algorithm

**New kernel:** `compute_csf_wetting(phi, flags, Fx, Fy, Fz, gamma_lu, theta_rad, FLAG_SOLID, FLAG_GAS, FLAG_FLUID, nx, ny, nz)`

1. For each fluid/interface cell with at least one solid neighbour:
   - Detect wall normal from solid occupancy (6-connectivity).
   - Compute \(\nabla\phi\) as today.
   - If \(|\nabla\phi| > \epsilon\), apply contact-angle correction to normal in the plane spanned by \(\nabla\phi\) and \(\hat{n}_\text{wall}\).
   - Compute κ from corrected normal (or use wall curvature approximation for flat substrate: κ ≈ 2 cos θ / dx near triple line).
2. Apply CSF force only in cells with \(f_l > f_l^\min\) and \(|\nabla\phi| > \epsilon\).
3. **Previously solidified bead** (`FLAG_SOLID` from `solidify_cooled_metal`) uses the **same** θ as substrate (same material).

**Alternative (simpler v1):** `apply_wall_contact_angle(phi, flags, ...)` ghost-cell φ redistribution at solid boundary before CSF (Ding & Spelt, 2008 style) — sets φ so local interface satisfies θ; then existing `compute_csf_tension` unchanged.

**Recommendation:** Start with **ghost-cell φ BC** (simpler to validate with static droplet); add explicit wall CSF if ghost-cell smears at coarse dx.

### 4.3 Files

| File | Change |
|------|--------|
| `kernels.py` | `apply_contact_angle_phi_bc`, optional `compute_csf_wetting` |
| `physics/free_surface.py` | Orchestrate φ BC before reinit |
| `physics/forces.py` | Export wetting CSF wrapper |
| `materials/schema` or `MaterialProps` | `contact_angle_deg`, optional `gamma_sl`, `gamma_sv` |
| `job.py` | `surface_wetting.contact_angle_deg` override |
| `twin.py` | `enable_wetting`, `contact_angle_deg`, `theta_rad` |
| `coupled_step.py` | Call φ BC after `reinitialize_phi`, before CSF |

### 4.4 Config schema

```yaml
# materials/validated/ER70S-6.v1.yaml (add)
surface:
  contact_angle_deg: 80.0      # molten steel on steel, illustrative
  # optional advanced:
  # gamma_sl_N_m: 1.72
  # gamma_sv_N_m: 1.95

# jobs/examples/bead_on_plate.yaml (optional override)
surface_wetting:
  contact_angle_deg: 75.0
simulation:
  enable_wetting: true
```

### 4.5 Validation

| Test | File | Acceptance |
|------|------|------------|
| Static sessile droplet | `validation/test_wetting_droplet.py` | Equilibrium cap height within 15% of Young–Laplace on flat plate (2D slice or axisymmetric check) |
| Toe angle sign | `validation/test_wetting_toe.py` | Contact line slope matches θ within 10° (visual or automated from φ field) |
| Laplace regression | `test_laplace.py` | Still passes with wetting off |
| Speed sweep | `validation/test_bead_aspect_speed.py` | At fixed θ, h/w decreases when travel speed ↑ (no refit) — qualitative gate |

### 4.6 Risks

- Contact angle on coarse grids (dx = 0.3 mm): capillary length \(\lambda_c = \sqrt{\gamma/(\rho g)} \approx 5\) mm for steel — need ≥ 3–5 cells across pool toe. Document minimum dx for wetting accuracy.
- θ hysteresis (advancing vs receding): defer to v2; use single θ first.

---

## 5. WP2 — Physical deposition inlet

### 5.1 Physics

Wire mass flux:

\[
\dot{m} = \rho \, \frac{\pi d_w^2}{4} \, v_\text{wire}
\]

Per droplet (frequency \(f_\text{drop}\)):

\[
m_\text{drop} = \dot{m} / f_\text{drop}, \quad V_\text{drop} = m_\text{drop} / \rho
\]

**Current bug:** `feed_wire` deposits into `FLAG_GAS` cells in a sphere **plus vertical column** (`in_column`), creating metal above the pool independent of surface physics.

**Target:** Deposit only into cells that are:
- `FLAG_FLUID` with \(f_l \ge f_l^\text{dep}\), **or**
- `FLAG_GAS` cells **directly adjacent** to the pool surface (one layer above interface), within a **footprint** of radius \(r_\text{foot}\) around arc centre.

Footprint radius (physics-based default):

\[
r_\text{foot} = \max\left( r_\text{drop}, \; w_\text{pool}/2 \right)
\]

where \(w_\text{pool}\) is estimated from recent telemetry or local hot-cell span. v1: use `sigma_cells` arc Gaussian radius or fixed multiple of wire diameter.

Droplet **centre** at wire tip: \((i_\text{arc}, j_\text{arc}, k_\text{tip})\) with \(k_\text{tip} = k_\text{arc} + k_\text{stickout\_cells}\) (see WP5; v1 use `arc_k + 1`).

**Enthalpy:** \(H = c_p \rho T_\text{drop} + L_f \rho\) with \(T_\text{drop}\) from stick-out preheat (WP5) or default `T_liquidus + ΔT_superheat`.

**Momentum:** Keep `feed_wire_momentum_impact` / `weld_forces.apply_droplet_impact` — applied only to deposited cells.

### 5.2 Algorithm

**Replace** `feed_wire` search logic:

```
REMOVE: in_column (vertical gas fill)
ADD:
  1. Build candidate list: gas cells k > k_arc with neighbour fluid OR fluid cells at interface
  2. Sort by distance to drop_cz (prefer pool surface)
  3. Convert gas→fluid until vol_acc >= target_vol
  4. If volume not placed, expand footprint radius (max 2 expansions) — NOT vertical column
  5. If still unplaced, log telemetry warning `deposition_overflow` (do not fill arbitrary gas)
```

**New kernel:** `feed_wire_surface(phi, flags, f_l, ...)` implementing the above.

**Optional:** distribute volume with Gaussian weights over footprint (smoother than first-come cells).

### 5.3 Files

| File | Change |
|------|--------|
| `kernels.py` | `feed_wire_surface` (deprecate column branch) |
| `physics/deposition.py` | Route to new kernel; `deposition_footprint_cells(twin)` |
| `physics/deposition_balance.py` | Unchanged mass formulas |
| `coupled_step.py` | Call new feed; track `deposition_overflow` in telemetry |
| `export/meta.py` | Export overflow count |

### 5.4 Config

```yaml
deposition:
  footprint_sigma_cells: null   # null → use twin.sigma_cells
  superheat_K: 500.0            # T_drop = T_liquidus + superheat_K
  min_liquid_fraction: 0.0      # allow deposit on interface gas neighbours only
  allow_footprint_expansion: true
  max_footprint_expansions: 2
```

### 5.5 Validation

| Test | Acceptance |
|------|------------|
| `test_mass_balance.py` | Ratio 0.95–1.05 (unchanged) |
| `test_deposition_no_column.py` | No new `FLAG_FLUID` more than `r_drop + 2` cells above local pool surface |
| `test_deposition_speed_aspect.py` | With WP1 on: crown height ↓ when footprint widens with pool width |

---

## 6. WP3 — Hydrostatic gravity

### 6.1 Physics

Full gravity body force on liquid (not only Boussinesq thermal):

\[
\mathbf{F}_g = \rho \, \mathbf{g} \, f_l
\]

In lattice units (consistent with existing force scaling):

\[
F_{z,\text{lu}} = -\rho_\text{lu} \, g_\text{lu} \, f_l
\]

where `g_lu` already exists on `WAAMTwin` (from 9.81 m/s²). **Sign:** +z upward in grid → gravity is \(-\rho g \hat{z}\).

Thermal Boussinesq (`add_buoyancy`) remains for natural convection; hydrostatic term is **additive**.

### 6.2 Algorithm

**New kernel:** `add_hydrostatic_gravity(Fz, f_l, flags, rho, g_lu, FLAG_SOLID, FLAG_GAS)`

Called after Marangoni, before or with buoyancy.

**Flag:** `enable_hydrostatic_gravity: true` (default **true** when `enable_vof` and `enable_csf_tension`).

### 6.3 Validation

| Test | Acceptance |
|------|------------|
| `test_hydrostatic_flat_pool.py` | Isothermal liquid puddle on flat plate: centre height ≤ edge height after settle steps |
| Capillary length check | Document \(Bo = \rho g L^2 / \gamma\); for L = 5 mm, Bo ≈ 0.1 — gravity is secondary to γ but matters for crown |

---

## 7. WP4 — Solidification–flow coupling

### 7.1 Physics

Bead profile is **locked** when the tail freezes. Current `solidify_cooled_metal` promotes `f_l < 0.02` and `T < T_solidus` to `FLAG_SOLID`. Gaps:

1. Solidification front may lag flow — liquid behind arc still mobile while toe already solid.
2. No **velocity zeroing** in newly solid cells (LBM may retain ux in SOLID via bounce-back only).
3. `enable_substrate_growth` runs solidify before next layer but single-bead crown forms in **same step** as deposition.

### 7.2 Enhancements

| ID | Feature | Implementation |
|----|---------|----------------|
| 4.1 | Mushy-zone velocity damping | Increase `C_darcy` or local τ as \(f_l \to 0\) (partially exists via `collide_*` + `f_l`) — verify and tune from `mu(T)` |
| 4.2 | Solidification front velocity | Optional Scheil-style: \(f_l(T)\) already from enthalpy; ensure `update_phase` runs **before** VOF advect |
| 4.3 | Post-freeze φ lock | In `solidify_cooled_metal`: set `phi = 1`, zero `ux,uy,uz` in newly SOLID cells |
| 4.4 | Bead height metric | Telemetry: `bead_height_mm` = max z of `FLAG_SOLID` deposited metal above `nz_solid` |

### 7.3 Step order adjustment

Ensure order (see §9):

```
phase_change → advect_phi → reinit_phi → contact_angle_bc →
solidify_cooled_metal (if substrate_growth OR continuous freeze) →
CSF + Marangoni + gravity → LBM
```

For **single bead on plate**, enable **continuous freeze** (`enable_bead_freeze: true`): call `solidify_cooled_metal` every step so crown accumulates as frozen SOLID behind arc.

### 7.4 Config

```yaml
simulation:
  enable_bead_freeze: true       # single-layer: freeze trail to SOLID
  enable_substrate_growth: true  # multi-layer: existing behaviour
```

### 7.5 Validation

| Test | Acceptance |
|------|------------|
| `test_two_layer_remelt.py` | Still passes |
| `test_bead_height_telemetry.py` | `bead_height_mm` monotonic during weld; stable after arc passes |
| `test_freeze_stops_flow.py` | max \|u\| in SOLID cells ≈ 0 |

---

## 8. WP5 — CTWD / stick-out resistance (multi-layer)

### 8.1 Physics

**Contact tube to work distance (CTWD)** and **electrode stick-out** affect:

1. Wire resistance preheat: \(P_\text{stick} = I^2 R_\text{stick}(L_\text{stick})\)
2. Effective arc voltage / melt rate balance
3. Droplet temperature at impact

Literature: Klaus et al. (2022) — short-circuit resistance correlates with CTWD; closed-loop Z control (2023).

### 8.2 Lumped model (v1)

**State variables** (on `WAAMTwin`):

- `ctwd_m` — contact tip to pool surface distance  
- `stickout_m` — wire extension below contact tip to arc  
- `torch_z_m` — absolute torch height (from path or layer stack)

**Resistance:**

\[
R_\text{stick} = \frac{\rho_e L_\text{stick}}{A_w}
\]

\(\rho_e\) ≈ 1.5×10⁻⁷ Ω·m (steel wire), \(A_w = \pi d_w^2/4\).

**Wire preheat power** (fraction of \(I^2 R\)):

\[
P_\text{preheat} = \eta_\text{stick} \, I^2 R_\text{stick}
\]

**Droplet enthalpy boost:** raise `T_drop` from stick-out energy balance (1D steady):

\[
\dot{m} c_p (T_\text{drop} - T_0) = P_\text{preheat}
\]

**CTWD update** each layer or from solidified height:

\[
\text{CTWD}_{n+1} = \text{CTWD}_\text{nominal} + (h_\text{bead,measured} - h_\text{layer,planned})
\]

Couple to existing `layer_height_mm` in `two_layer.yaml` and `torch_path` z coordinates.

### 8.3 Coupling to Lorentz (optional v2)

Extend `solve_lorentz` with 1D wire path in series with 3D pool conductivity — pool `sigma_elec` already exists. v1: preheat only, no change to pool J solve.

### 8.4 Files

| File | Change |
|------|--------|
| `physics/electrical_stickout.py` | New module: R_stick, T_drop, CTWD update |
| `twin.py` | `ctwd_m`, `stickout_m`, `rho_e`, `eta_stick` |
| `job.py` | `process.ctwd_mm`, `process.stickout_mm`, `electrical.rho_e_ohm_m` |
| `coupled_step.py` | Update CTWD after `solidify`; set `T_drop` before `feed_wire` |
| `solvers/layer_scheduler.py` | Optional: interpass + Z step from `layer_height_mm` |

### 8.5 Validation

| Test | Acceptance |
|------|------------|
| `test_stickout_preheat.py` | T_drop increases with stickout length |
| `test_ctwd_layer_drift.py` | Open-loop vs planned layer height: CTWD error grows; with correction, bounded |
| `test_two_layer.yaml` integration | Layer 2 pool depth differs from layer 1 when CTWD changes |

---

## 9. Updated `coupled_step` order

```
 1. clear_forces(Fx, Fy, Fz)
 2. [WP5] update_ctwd_from_bead_height()          [if enable_ctwd]
 3. arc.inject_heat(H)                            [if welding]
 4. [WP2] deposition.feed_wire_surface()         [on droplet schedule]
 5. [WP2] weld_forces.apply_droplet_impact()      [on droplet schedule]
 6. deposition.inject_tracers()                    [on droplet schedule]
 7. thermal.advect_diffuse(H, T, u)
 8. thermal.apply_boundary_losses + enthalpy cap
 9. phase_change.update(H, T, f_l)
10. thermal.update_T_max
11. [WP4] solidify_cooled_metal / remelt           [substrate_growth / bead_freeze]
12. free_surface.advect_phi(phi, u)               [if enable_vof]
13. free_surface.reinitialize_phi(phi)
14. [WP1] free_surface.apply_contact_angle_bc(phi)
15. free_surface.update_flags_from_phi(phi, flags)
16. [WP5] electrical_stickout.set_T_drop()        [before next deposition event]
17. forces.compute_csf_tension(phi)              [if enable_csf_tension]
18. forces.marangoni(Fx, Fy, Fz)
19. [WP3] forces.add_hydrostatic_gravity(Fz)       [if enable_hydrostatic_gravity]
20. forces.add_buoyancy(Fz)                       [thermal Boussinesq]
21. [welding] lorentz, gas_shear, arc_pressure, recoil
22. lbm.collide_srt | collide_mrt
23. lbm.stream
24. deposition.advect_tracers
25. [WP4] telemetry: bead_height_mm, ctwd_m
26. kernels.snapshot_forces(Fx_snap, ...)
27. grid.swap_buffers()
```

---

## 10. Material & job schema summary

### 10.1 Material YAML (`materials/validated/*.yaml`)

```yaml
constants:
  gamma_0: 1.8
  dgamma_dT: -4.3e-4
  # ... existing ...

surface:
  contact_angle_deg: 80.0

electrical:                    # WP5
  rho_e_ohm_m: 1.5e-7
  eta_stick: 0.85              # fraction of I²R to wire preheat
```

### 10.2 Job YAML

```yaml
simulation:
  enable_wetting: true
  enable_hydrostatic_gravity: true
  enable_bead_freeze: true
  enable_ctwd: false           # WP5; true for two_layer jobs

surface_wetting:
  contact_angle_deg: 75.0      # overrides material

deposition:
  superheat_K: 500.0
  footprint_sigma_cells: null

process:
  ctwd_mm: 15.0
  stickout_mm: 12.0
  wire_feed_m_min: 8
  travel_speed_mm_s: 8

layer_height_mm: 1.2           # multi-layer planned step (existing)
```

### 10.3 Telemetry additions (`get_telemetry()`)

| Key | Unit | Description |
|-----|------|-------------|
| `bead_height_mm` | mm | Max deposited SOLID above substrate |
| `toe_angle_deg` | deg | Estimated from φ slope at substrate contact (WP1+) |
| `deposition_overflow` | bool / count | Volume not placed on footprint |
| `ctwd_mm` | mm | WP5 |
| `T_drop_K` | K | Droplet entry temperature |

---

## 11. VTK / diagnostics

| Field | When |
|-------|------|
| `Contact_Angle_target_deg` | Scalar on meta JSON |
| `Bead_Height_mm` | Per-frame telemetry CSV |
| `phi` with wetting BC | Existing volume export |
| Surface `Toe_Angle` | Optional derived on `.vtp` from φ gradient at substrate |

Update `docs/DIAGNOSTICS_AND_VTK_SPEC.md` §Layer A when WP1 ships.

---

## 12. Implementation phases & task IDs

### Phase G1 — Wetting + deposition (weeks 1–2)

| Task | WS | Files | Gate |
|------|-----|-------|------|
| G1.1 `apply_contact_angle_phi_bc` | F | `kernels.py`, `free_surface.py` | `test_wetting_droplet` |
| G1.2 `enable_wetting` + job/material schema | G | `twin.py`, `job.py`, materials | `test_job_parity` |
| G1.3 `feed_wire_surface` remove column | G | `kernels.py`, `deposition.py` | `test_deposition_no_column` |
| G1.4 Wire `bead_on_plate.yaml` defaults | G | job example | manual review |

### Phase G2 — Gravity + freeze (week 3)

| Task | WS | Files | Gate |
|------|-----|-------|------|
| G2.1 `add_hydrostatic_gravity` | F | `kernels.py`, `forces.py` | `test_hydrostatic_flat_pool` |
| G2.2 `enable_bead_freeze` + velocity zero | G | `coupled_step.py`, `solidify` | `test_freeze_stops_flow` |
| G2.3 `bead_height_mm` telemetry | G | `twin.py` | `test_bead_height_telemetry` |

### Phase G3 — Validation matrix (week 4)

| Task | WS | Files | Gate |
|------|-----|-------|------|
| G3.1 Speed sweep aspect ratio | H | `test_bead_aspect_speed.py` | h/w monotonic vs travel speed |
| G3.2 Update `reference_case_ER70S6.md` | docs | bead height targets | review |
| G3.3 Deprecate geometry calibration knobs in docs | docs | README, MATERIALS | review |

### Phase G4 — CTWD / multi-layer (weeks 5–6)

| Task | WS | Files | Gate |
|------|-----|-------|------|
| G4.1 `electrical_stickout.py` | G | new module | `test_stickout_preheat` |
| G4.2 CTWD from bead height | G | `coupled_step`, `two_layer.yaml` | `test_ctwd_layer_drift` |
| G4.3 Integrate with `layer_height_mm` path | G | `job.py`, path driver | `test_two_layer_remelt` |

---

## 13. Acceptance criteria (physics gates)

**Release gate for G1+G2** (replace single-scenario pool-only fit):

| Metric | Method | Criterion |
|--------|--------|-----------|
| Mass balance | `test_mass_balance` | 0.95 ≤ ratio ≤ 1.05 |
| Sessile droplet | WP1 test | Cap shape within 15% theory |
| Crown height | bead_on_plate @ 8 mm/s | ≤ 3 mm reinforcement (adjust after first GPU run) |
| Aspect ratio trend | 3 speeds, fixed θ | h/w decreases as travel speed ↑ |
| Wetting angle | φ slope at toe | Within 15° of input θ |
| No column deposit | WP2 test | Zero fluid cells > 2 cells above interface |

**Explicitly not a gate:** matching `model_reference` pool_depth_mm alone without bead height.

---

## 14. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Coarse dx smears contact angle | Document min dx; recommend `standard` preset for bead geometry studies |
| WP1 + WP2 reduce pool depth | Re-verify thermal tests; do not refit η for geometry |
| Hydrostatic + CSF stiffness | Sub-stepping or limit g_lu; monitor LBM stability |
| CTWD model too crude | v1 preheat only; v2 couples to Lorentz |
| Performance | Wetting BC + new deposit search: profile on `standard` grid |

---

## 15. References

1. Brackbill, Kothe & Zemach — CSF method, *J. Comput. Phys.* 1992.  
2. Ding & Spelt — Wetting BC for VOF, *J. Comput. Phys.* 2008.  
3. Zhao et al. — GMAW-WAAM heat transfer, flow, geometry, *Welding in the World* 2021. [doi:10.1007/s40194-021-01123-1](https://doi.org/10.1007/s40194-021-01123-1)  
4. ER70S-6 experimental + CFD + RSM, *Proc. IMechE* 2025. [doi:10.1177/09544062251351162](https://doi.org/10.1177/09544062251351162)  
5. Ogino et al. — Stable deposited height GMAW-AM, *Appl. Sci.* 2020. [MDPI](https://www.mdpi.com/2076-3417/10/12/4322)  
6. Klaus et al. — CTWD sensing via short-circuit resistance, *Int J Adv Manuf Technol* 2022. [Springer](https://link.springer.com/article/10.1007/s00170-022-08805-0)  
7. Closed-loop Z from process signal, *Int J Adv Manuf Technol* 2023.  
8. ETH deposition / footprint locus model — [research-collection.ethz.ch](https://www.research-collection.ethz.ch/server/api/core/bitstreams/067f3081-b856-4fe9-aa57-df28ff6afe8e/content)  
9. WAAM bead geometry review — Pertanika JST 2024.  

---

## 16. Cross-references

- `docs/WAAM_TWIN_V2_EXECUTION_PLAN.md` §7 step order — **supersede** with §9 when implementing  
- `docs/DIAGNOSTICS_AND_VTK_SPEC.md` — surface / κ exports  
- `docs/validation/reference_case_ER70S6.md` — add bead height column after G3  
- `jobs/examples/bead_on_plate.yaml` — enable flags after G1  
- `jobs/examples/two_layer.yaml` — WP5 integration target  

---

*End of spec v1.0*
