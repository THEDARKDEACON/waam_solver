# Physics Force Correctness Spec

**Status:** Implementation brief (fix plan)  
**Audience:** Kernel / `coupled_step` / validation authors  
**Depends on:** `WAAM_WELD_POOL_PHYSICS_CENTRE.md`, `BEAD_GEOMETRY_PHYSICS_SPEC.md`, `solvers/coupled_step.py`, `kernels.py`  
**Goal:** A continuum WAAM melt-pool simulator whose **live force balance matches the documented equations**, with every Physics Centre force present, additive, and unit-consistent.

---

## 1. Problem statement

Coverage of weld-pool mechanisms is largely complete (heat source, enthalpy–porosity, VOF/CSF, Marangoni, buoyancy, gravity, Lorentz, arc pressure, recoil, gas shear, droplet mass/momentum). **Equation fidelity in the assembled timestep is not.**

Primary defects found in audit (2026-07):

| ID | Defect | Impact |
|----|--------|--------|
| **FC-1** | Marangoni uses `Fx = …` (overwrite), not `+=` | Wipes CSF capillary force on the interface |
| **FC-2** | Marangoni zeros `F` on non-interface fluid cells | Erases any prior surface force; brittle ordering |
| **FC-3** | CSF curvature stencil ≠ Brackbill \(\kappa=-\nabla\cdot\hat n\) | Wrong capillary pressure / toe |
| **FC-4** | Arc pressure is fixed `Pa`, not Lin–Eagar \(I^2\) | Process-window drift; penetration under/over |
| **FC-5** | “Goldak” is energy-normalized shared-\(\sigma\) ellipsoid | Not textbook Goldak spatial PDF |
| **FC-6** | Wetting uses \(\kappa=2\cos\theta\) + empirical \(\sin\theta\) drive | Toe not Young-faithful |
| **FC-7** | Recoil CC missing accommodation factor \(\approx 0.54\) | Optional; over-strong if recoil enabled |
| **FC-8** | Lorentz / gas shear often OFF in production jobs | Force catalogue incomplete in practice |
| **FC-9** | Soft failure (overflow / Jacobi) never aborts | Silent wrong physics |
| **FC-10** | `lorentz_body_force_peak_N_m3` uses \(I^2/r^2\) (pressure scale) | Wrong diagnostic units |

**Success criterion:** With a single “full physics” job preset, every force in §3 is active, additive, and passes the §7 acceptance tests against analytic or literature-scaled checks.

---

## 2. Non-goals

- Full Tanaka–Lowke arc-plasma MHD (arc enters as boundary fluxes only).
- Grain structure, residual-stress FEA, microsegregation.
- True CMT wire retract / short-circuit detachment CFD (keep phenomenological droplets; improve scaling only).
- Replacing LBM with FVM.

---

## 3. Force & flux catalogue (must all contribute)

Target interface force density (lattice Guo acceleration after conversion):

\[
\mathbf{F}_{\mathrm{total}}
=
\underbrace{\gamma\,\kappa\,\nabla\phi}_{\mathrm{CSF}}
+
\underbrace{\frac{\mathrm{d}\gamma}{\mathrm{d}T}\,\nabla_s T\,|\nabla\phi|}_{\mathrm{Marangoni}}
+
\underbrace{\mathbf{n}\,p_{\mathrm{arc}}}_{\mathrm{arc\,pressure}}
+
\underbrace{\mathbf{n}\,p_{\mathrm{recoil}}}_{\mathrm{recoil}}
+
\underbrace{\boldsymbol{\tau}_{\mathrm{gas}}}_{\mathrm{gas\,shear}}
+
\underbrace{\mathbf{n}\,p_{\mathrm{drop}}}_{\mathrm{droplet\,impact}}
+
\underbrace{\mathbf{J}\times\mathbf{B}}_{\mathrm{Lorentz}}
+
\underbrace{\rho\mathbf{g}\,f_l}_{\mathrm{hydrostatic}}
+
\underbrace{-\rho g\beta(T-T_{\mathrm{ref}})\,\hat{\mathbf{z}}\,f_l}_{\mathrm{Boussinesq\ (sign:\ +z\ up\ \Rightarrow\ +\rho g\beta\Delta T\ in\ code)}}
\]

plus mushy Darcy drag inside collision (not a free-surface force).

Heat / mass fluxes (not body forces, but required for consistent pool):

| Flux | Equation target |
|------|-----------------|
| Arc heat | \(\int \dot q\,\mathrm{d}V=\eta I V\,\Delta t\) per step (already OK) |
| Goldak PDF | §5.2 |
| Droplet mass | \(\dot m=\rho\,(\pi/4)d_w^2 v_{\mathrm{wire}}\) |
| Droplet enthalpy | \(H_{\mathrm{drop}}=\rho c_p T_{\mathrm{drop}}+\rho L_f\) |
| Stick-out preheat | \(P=\eta_{\mathrm{stick}} I^2 R_{\mathrm{stick}}\) → \(T_{\mathrm{drop}}\) |

---

## 4. Hard contracts (all kernels)

### 4.1 Force assembly contract

1. **`clear_forces` once** at the start of the force block.
2. **Every force kernel uses `+=` only.** Never assign `Fx=…` on cells that may hold other forces.
3. **Never zero `F`** except inside `clear_forces`.
4. Gate by physics, not by wiping:
   - CSF / Marangoni: require \(|\nabla\phi|>\varepsilon\) and \(f_l>f_l^{\min}\) (suggested \(0.05\)).
   - Body forces: require fluid/mushy, not `FLAG_GAS`.
5. Order (matches Physics Centre §16, surface then body):

```
clear_forces
→ CSF (+ wetting)
→ Marangoni
→ gas shear          [if enabled]
→ arc pressure       [if welding]
→ recoil             [if enabled]
→ droplet impact P   [if droplet this step]
→ hydrostatic gravity
→ Boussinesq buoyancy
→ Lorentz J×B        [if enabled]
→ collide (Darcy inside) → stream
```

Droplet **momentum** injection into `f` / `u` may remain with deposition (before or after thermal); droplet **pressure** belongs in the surface-force block above.

### 4.2 Lattice unit contract

With \(\rho_{lu}\approx 1\) Guo forcing:

\[
\mathbf{a}_{lu}=\frac{\mathbf{f}_{\mathrm{phys}}}{\rho}\,\frac{\Delta t^2}{\Delta x}
\qquad
\bigl[\mathrm{lu}/\mathrm{ts}^2\bigr]
\]

| Physical quantity | Lattice conversion already used |
|-------------------|----------------------------------|
| Marangoni | \(d\gamma_{lu}=(d\gamma/dT)\,(dt^2)/(\rho\,dx^3)\) then \(F=d\gamma_{lu}\,\nabla_s T_{lu}\,|\nabla\phi|_{lu}\) |
| CSF \(\gamma\) | \(\gamma_{lu}=\gamma_0\,(dt^2)/(\rho\,dx^3)\) then \(F=\gamma_{lu}\,\kappa_{lu}\,\nabla\phi_{lu}\) |
| Pressure → \(F_z\) | \(a_{lu}=P/(\rho\,dx)\cdot dt^2/dx\) (one-cell support) |
| Lorentz | \(a_{lu}=(J\times B)/\rho\cdot dt^2/dx\) |

**Lock these in comments + a unit-test that checks dimensional groups**, so future edits cannot drop a \(1/dx\).

### 4.3 Feature flags vs “full physics”

Introduce job preset / simulation key:

```yaml
simulation:
  physics_tier: full   # thermal | flow | full
```

| Tier | Enables |
|------|---------|
| `thermal` | heat + phase change only (validation) |
| `flow` | + VOF + CSF + Marangoni + buoyancy + gravity + Darcy |
| `full` | + Lorentz + arc pressure (Lin–Eagar) + gas shear + droplet impact + wetting + bead freeze; recoil optional via `enable_recoil` |

`jobs/examples/bead_on_plate.yaml` must migrate to `physics_tier: full` (recoil default **false** for conduction-mode WAAM).

---

## 5. Per-mechanism fix specs

### 5.1 FC-1 / FC-2 — Marangoni additive (P0)

**Files:** `kernels.compute_marangoni_force`, `compute_marangoni_force_variable`

**Change:**

```text
# BEFORE (broken)
Fx[i,j,k] = scale * dTs_x

# AFTER
if grad_phi_mag < eps or f_l[i,j,k] < fl_min:
    continue          # do not touch F
Fx[i,j,k] += scale * dTs_x
...
```

Remove the branches that set `F=0` for solid/gas (those cells are already skipped) and for low \(|\nabla\phi|\`.

**Acceptance:**

1. Unit test: apply CSF only → record \(F_{\mathrm{csf}}\); apply CSF then Marangoni → \(F_{\mathrm{csf+m}}\); assert \(\|F_{\mathrm{csf+m}}-F_{\mathrm{csf}}-F_{\mathrm{m}}\|_\infty < 10^{-10}\).
2. Existing `test_marangoni_cell` still passes (direction vs sign of \(d\gamma/dT\)).
3. `test_laplace` still nonzero after Marangoni with isothermal \(T\) (Marangoni ~0, CSF preserved).

---

### 5.2 FC-3 — Brackbill curvature (P0)

**Files:** `kernels.compute_csf_tension`

**Target:**

\[
\hat n=\frac{\nabla\phi}{|\nabla\phi|},\qquad
\kappa=-\nabla\cdot\hat n,\qquad
\mathbf F=\gamma\,\kappa\,\nabla\phi
\]

**Algorithm:**

1. Compute \(\nabla\phi\) with central differences (existing).
2. At each neighbour face/cell, reconstruct \(\hat n\), then discrete divergence of \(\hat n\) (standard CSF / balanced-force stencil). Prefer a known scheme (e.g. Brackbill 1992 or balanced-force CSF as in Francois et al.) over the current asymmetric \(\phi\) second difference.
3. Near domain faces: either one-sided \(\hat n\) or skip cells lacking a full stencil (current skip is OK if documented).
4. **Do not** multiply by an extra \(|\nabla\phi|\) beyond \(\mathbf F=\gamma\kappa\nabla\phi\).

**Wetting (replace FC-6 crude wall κ):**

- Prefer Ding–Spelt ghost \(\phi\) BC (already present) **plus** wall-normal correction of \(\hat n\):

\[
\hat n_{\mathrm{eff}}=\sin\theta\,\hat n_{\mathrm{wall}}+\cos\theta\,\hat t
\]

- Remove the ad-hoc lateral \(\sin\theta\) force once ghost-φ + normal correction are validated.
- Keep `enable_wetting` gate.

**Acceptance:**

1. **Laplace law:** spherical drop in zero-g, \(\Delta P = 2\gamma/R\) (3D) or \(\gamma/R\) (2D cylinder) within 15% at \(dx/R\le 1/8\).
2. **Static sessile drop:** equilibrium contact angle within \(10^\circ\) of prescribed \(\theta\) (`test_wetting_droplet` tighten).
3. Spurious current \(\mathrm{Re}_{spurious}\) documented; must not grow unbounded over 5k steps.

---

### 5.3 FC-4 — Arc pressure Lin–Eagar (P0)

**Files:** `physics/weld_forces.py` (new helper), `coupled_step.py`, `twin.py`, job schema

**Default peak pressure:**

\[
p_0(I)=\frac{\mu_0 I^2}{4\pi^2 \sigma_p^2}
\qquad
p(r)=p_0\exp\!\Bigl(-\frac{r^2}{2\sigma_p^2}\Bigr)
\]

with \(\sigma_p\) defaulting to arc heat \(\sigma\) (cells→m) unless `arc_physics.pressure_sigma_mm` set.

**API:**

```yaml
arc_physics:
  pressure_model: lin_eagar   # or constant
  pressure_pa: 500.0          # used only if constant
  pressure_sigma_mm: null     # null → use arc sigma
```

Map to lattice with existing `_wf_pressure_to_Fz_lu`. Apply along **interface normal** (prefer \(\mathbf F=-p\,\hat n\,|\nabla\phi|\) CSF-style) instead of \(F_z\)-only when VOF is on; keep \(F_z\)-only fallback if \(|n_z|\) tiny.

**Acceptance:**

1. Doubling \(I\) → \(p_0\) ×4 (unit test).
2. Downward / inward surface force under arc (`test_arc_pressure` extended).
3. With Marangoni+CSF on, arc pressure still measurable (additive).

---

### 5.4 FC-5 — Goldak spatial fidelity (P1)

**Files:** `kernels.inject_goldak_heat`, `physics/arc.py`, job `goldak:` block

**Target (Goldak 1984), front/rear:**

\[
q_{f,r}=\frac{6\sqrt{3}\,f_{f,r}\,Q}{\pi\sqrt{\pi}\,a_{f,r}\,b\,c}
\exp\!\Bigl(-3\frac{x^2}{a_{f,r}^2}-3\frac{y^2}{b^2}-3\frac{z^2}{c^2}\Bigr)
\]

with \(f_f+f_r=2\), \(Q=\eta IV\).

**Keep:** two-pass energy renormalization so intercepted metal receives exactly \(\eta Q\Delta t\) (numerical necessity on discrete grids).

**Job fields (mm → cells):**

```yaml
goldak:
  ff: 0.6
  fr: 1.4          # enforce ff+fr=2 in loader (warn + renormalize)
  a_front_mm: ...
  a_rear_mm: ...
  b_mm: ...
  c_mm: ...        # depth
```

Deprecate ambiguous `depth_front_mm` / shared `sigma` or map them explicitly in comments.

**Acceptance:**

1. Loader asserts \(f_f+f_r\approx 2\).
2. Front/rear asymmetry visible in \(T_{\max}\) wake vs lead (smoke).
3. Total enthalpy rise per step = \(\eta Q\Delta t\) ± 1%.

---

### 5.5 FC-7 — Recoil accommodation (P2)

**Files:** `apply_vapor_recoil_clausius_clapeyron`

\[
p_{\mathrm{recoil}}=C_{\mathrm{acc}}\,P_{\mathrm{sat}}(T),\qquad C_{\mathrm{acc}}\approx 0.54
\]

(config: `advanced_physics.recoil_accommodation: 0.54`). Default recoil **off** for WAAM conduction mode.

**Acceptance:** At \(T\le T_b\), force = 0; at \(T>T_b\), \(p\) matches formula within roundoff.

---

### 5.6 Lorentz readiness (P0 config + P1 robustness)

**Already structurally correct** (Ampère \(B_\theta\), \(J=-\sigma\nabla\phi\), Guo scale). Fixes:

1. Enable under `physics_tier: full`.
2. Fail or mark telemetry if `_lorentz_unconverged` exceeds threshold in `strict_mode`.
3. Fix `lorentz_body_force_peak_N_m3` → use \(\mu_0 I^2/(2\pi^2 r^3)\) body-force scale (as in `surfactant.lorentz_reference_accel_m_s2`), or rename to pressure.

**Acceptance:** `test_lorentz_physical_scale` + force ablation: Lorentz on vs off changes penetration directionally (deeper when on), Cho & Na consistent.

---

### 5.7 Gas shear (P1)

Keep \(\tau\sim\tfrac12\rho_g v_{\mathrm{jet}}^2 C_\tau\) tangential to surface (existing). Enable in `full` tier. Validate nonzero tangential \(F\) under jet with isothermal pool.

---

### 5.8 Droplet impact (P1)

Keep phenomenological \(v_{\mathrm{impact}}\) and \(p\sim E_k/A\). Ensure:

1. Momentum kernel and pressure kernel both `+=` / distribution update without clearing surface forces.
2. Impact pressure applied in the **same** force block as arc pressure (after clear+CSF+Marangoni), not only inside deposition in a way that gets cleared.

**Acceptance:** Cho & Na-style smoke: droplet-only case produces dominant vertical velocity vs buoyancy-only.

---

### 5.9 Surfactant (P2)

**Models** (`physics/surfactant.py`, material YAML `surfactant.model`):

1. **heiple** (default) — static S-ppm scale of base \(d\gamma/dT\).
2. **sahoo** — Sahoo/DebRoy/McNallan (1988) Fe–S:

\[
\gamma(T,a_S)=\gamma_m-A(T-T_m)-RT\Gamma_s\ln(1+K a_S),\quad
K=k_1\exp(-\Delta H^0/RT)
\]

with analytic \(d\gamma/dT(T,a_S)\). Activity \(a_S\approx\mathrm{wt\%\,S}=\mathrm{ppm}/10^4\).
Loader installs a dense \(d\gamma/dT\) table for `use_material_tables` (local in \(T\); \(a_S\) uniform — no solute advection).

**Acceptance:** low-S near liquidus \(d\gamma/dT<0\); high-S near liquidus \(d\gamma/dT>0\); high-\(T\) recovery toward negative (`test_surfactant_dgamma`).

---

## 6. `coupled_step` / job / twin work packages

| WP | Priority | Deliverable |
|----|----------|-------------|
| **WP-A** | P0 | FC-1/2 Marangoni `+=` + no zeroing — **DONE** |
| **WP-B** | P0 | FC-3 Brackbill CSF + Laplace test — **DONE** (`κ=−∇·n̂`, `test_laplace` ≤15%) |
| **WP-C** | P0 | Force-assembly order lock + additive regression — **DONE** (`test_force_additivity`) |
| **WP-D** | P0 | Lin–Eagar arc pressure + `physics_tier: full` job — **DONE** |
| **WP-E** | P0 | Enable Lorentz + gas shear in full tier; strict_mode gates — **DONE** |
| **WP-F** | P1 | Goldak PDF parameters — **DONE** (`a,b,c` cells + `ff+fr=2`, `test_goldak_energy`) |
| **WP-G** | P1 | Wetting normal correction; remove sinθ hack — **DONE** (Young n̂ in CSF; `test_wetting_wall_csf`) |
| **WP-H** | P2 | Recoil accommodation; surfactant local γ — **DONE** (`C_acc≈0.54`; Sahoo/DebRoy `model: sahoo`) |
| **WP-I** | P1 | Docs: update Physics Centre “implementation notes” pointing here — **DONE** |

**Recommended order:** A → C → B → D → E → F → G → H.

---

## 7. Validation matrix (must gate merges)

| Test | Asserts |
|------|---------|
| `test_force_additivity` (**new**) | CSF + Marangoni + buoyancy superposition |
| `test_laplace` (**tighten**) | \(\Delta P\) vs \(\gamma/R\) |
| `test_marangoni_cell` | Flow direction vs \(d\gamma/dT\) sign |
| `test_arc_pressure` | Lin–Eagar \(I^2\); downward interface load |
| `test_lorentz_physical_scale` | Order-of-magnitude \(J\times B\) |
| `test_hydrostatic_gravity` | Crown sag / Bond |
| `test_wetting_droplet` | \(\theta\) within 10° |
| `test_mass_balance` | Wire mass vs deposited |
| `test_goldak_energy` (**new**) | \(\Delta H=\eta Q\Delta t\) |
| Ablation job (tool) | Cho & Na style: droplet / Marangoni / EMF / buoyancy on-off velocity report |

CI: core suite always; `WAAM_FULL_VALIDATION=1` includes ablation + Laplace tolerance.

---

## 8. Telemetry / strict mode

Extend `get_telemetry()`:

```text
force_diagnostics:
  f_csf_max, f_marangoni_max, f_arc_max, f_lorentz_max,
  f_buoyancy_max, f_gravity_max, f_gas_shear_max, f_recoil_max
mass_balance_ratio
lorentz_unconverged_count
deposition_overflow_count
physics_tier
```

`strict_mode: true` (job or env `WAAM_STRICT=1`):

- Abort if `mass_balance_ratio ∉ [0.95, 1.05]` after N droplets.
- Abort if Lorentz unconverged streak > K.
- Abort if any force diagnostic is NaN.

---

## 9. Reference equations (copy into kernel docstrings)

**Marangoni (CSF volume form):**
\(\mathbf F_M=(d\gamma/dT)\,(\mathbf I-\hat n\hat n^T)\nabla T\,|\nabla\phi|\)

**CSF:**
\(\mathbf F_\gamma=\gamma\kappa\nabla\phi,\ \kappa=-\nabla\cdot\hat n\)

**Boussinesq (+z up):**
\(F_z=+\rho g\beta(T-T_{\mathrm{ref}})f_l\)

**Hydrostatic:**
\(\mathbf F_g=\rho\mathbf g f_l\) with \(g_z=-g\)

**Lin–Eagar arc pressure:**
\(p_0=\mu_0 I^2/(4\pi^2\sigma_p^2)\)

**Lorentz:**
\(\nabla\cdot(\sigma\nabla\phi_e)=S_I,\ \mathbf J=-\sigma\nabla\phi_e,\ B_\theta=\mu_0 I_{\mathrm{enc}}/(2\pi r),\ \mathbf F=\mathbf J\times\mathbf B\)

**Recoil:**
\(P_{\mathrm{sat}}=P_{\mathrm{ref}}\exp[\frac{L_v}{R}(\frac1{T_b}-\frac1T)],\ p_{\mathrm{rec}}=0.54\,P_{\mathrm{sat}}\)

**Darcy:**
\(\mathbf u\leftarrow\mathbf u/\bigl(1+C(1-f_l)^2/(f_l^3+\varepsilon)\bigr)\)

---

## 10. External literature (fix authority)

1. Brackbill et al. (1992) — CSF  
2. Oreper & Szekely (1984) — Marangoni + Lorentz  
3. Kou (2003) — weld pool flow taxonomy  
4. Lin & Eagar (1986) — arc pressure  
5. Goldak et al. (1984) — double ellipsoid  
6. Voller & Prakash (1987) — enthalpy–porosity  
7. Heiple & Roper (1982); Mills et al. (1998) — surfactant  
8. Cho & Na (2021) — GMAW force ablation ranking  
9. Matsunawa & Semak (1997) — recoil  
10. Wang et al. (2003) — metal transfer  

Repo anchors: `docs/WAAM_WELD_POOL_PHYSICS_CENTRE.md`, `docs/weld_pool_physics.md`.

---

## 11. Definition of done

- [x] WP-A…E merged; `physics_tier: full` is default for `bead_on_plate`
- [x] Additivity + Laplace + Lin–Eagar unit tests green in CI
- [x] Force diagnostics in telemetry; strict mode documented
- [x] README “physics” section links this spec
- [x] Macrograph gate: crown/penetration move in the correct direction under Lorentz on/off and S-ppm high/low (even if absolute mm still calibrating) — `test_force_direction_gate`

---

## 12. Immediate first patch (smallest merge)

1. Marangoni `+=` + remove force zeroing (FC-1/2). **DONE**
2. New `validation/test_force_additivity.py`. **DONE** (wired in `run_all`)
3. Re-order force assembly in `coupled_step` to match §4.1. **DONE**
   (CSF → Marangoni → gas shear → arc pressure → recoil → hydrostatic →
   buoyancy → Lorentz)

Estimate: <1 day. Unblocks all subsequent free-surface fidelity work.
