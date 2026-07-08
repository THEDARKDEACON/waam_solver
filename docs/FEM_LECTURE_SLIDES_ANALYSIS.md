# Analysis: FEM Weld-Pool Lecture Slides vs `waam_twin`

This document reviews a set of lecture slides on **finite-element modeling of heat transfer and fluid flow in welding** and assesses what is useful, what is mismatched, and what should *not* be adopted wholesale for the current `waam_twin` / `waam_solver` project.

**Audience:** developers extending the digital twin, calibration, validation, or thesis documentation.

**Date:** 2026-07-08

---

## Executive summary

The slides describe a **classical coupled FEM weld-pool model** in the tradition of Kou, DebRoy, and related GTA/GMAW CFD work:

- 3D **8-node brick elements**
- **Galerkin** weighted-residual discretization
- **Penalty method** for incompressibility
- **Quasi-steady** or moving-frame thermal formulation
- Coupled **energy + momentum** matrix systems
- Analytical **electromagnetic (Lorentz)** body-force expressions
- **Marangoni** boundary conditions on a **flat** free surface
- **Banded / frontal / iterative** sparse linear solvers

Our project (`waam_twin`) is **not** that architecture. It is a **GPU grid-based digital twin**:

- **Taichi LBM** for incompressible flow
- **Enthalpy–porosity** solidification
- **VOF** (`φ`) for free-surface topology
- **Goldak / Gaussian** arc heat sources
- **Explicit time marching** on a structured Cartesian grid
- Optional **Lorentz, recoil, gas shear, droplet deposition**

### Verdict

| Category | Assessment |
|----------|------------|
| **Physics concepts** | Very useful — confirms which forces and BCs matter |
| **Validation targets** | Useful — pool shape, Marangoni direction, HAZ, symmetry assumptions |
| **Numerical method** | Mostly **not portable** — FEM assembly ≠ LBM/VOF |
| **Implementation workflow** | **Not applicable** — mesh reordering, banded solvers, frontal elimination |
| **Strategic value** | High as a **reference theory layer**, low as a **code blueprint** |

**Bottom line:** treat the slides as a **physics and validation syllabus**, not a roadmap to rewrite the solver.

---

## What the slides represent

Across the image set, the lecture builds a complete **FEM-based weld pool simulator**:

1. **Governing PDEs** — conservation of mass, momentum (Navier–Stokes), and energy
2. **Constitutive relations** — Stokes stress, Boussinesq buoyancy
3. **Boundary conditions** — symmetry, no-slip solid–liquid interface, flat-surface Marangoni shear, convective/radiative losses, Gaussian arc heat input
4. **Surface tension physics** — temperature- and solute-dependent \( \partial\gamma/\partial T \)
5. **Electromagnetic forces** — analytical \(F_{em}^{x,y,z}\) from arc current
6. **FEM discretization** — shape functions, element matrices, penalty continuity enforcement
7. **Time discretization** — implicit schemes assembling effective stiffness systems
8. **Solver infrastructure** — banded storage, frontal method, iterative LIS

This is a mature **academic/industrial weld-pool CFD** formulation, typically used for:

- understanding Marangoni vs Lorentz dominance
- predicting pool width/depth trends
- process-parameter studies in 2D/3D symmetric domains

It is **not** a bead-on-plate transient WAAM deposition twin with wire feed, droplets, and layer buildup.

---

## What `waam_twin` currently is

From `README.md`, `coupled_step.py`, and `docs/weld_pool_physics.md`:

| Aspect | `waam_twin` today |
|--------|-------------------|
| Discretization | Structured 3D grid, lattice units |
| Flow solver | LBM collide + stream (SRT/MRT) |
| Solidification | Enthalpy + liquid fraction |
| Free surface | VOF `φ`, CSF Marangoni, wetting contact angle |
| Heat source | Goldak double-ellipsoid or Gaussian2D |
| Arc/body forces | Lorentz (optional), recoil, gas shear, droplet impact |
| Deposition | Wire feed, droplet transfer modes, bead freeze |
| Time integration | Explicit per-step orchestration |
| Output | VTK bundles, telemetry, ParaView sequences |
| Stated non-goals | Grain structure, residual-stress FEA, powder DEM |

The project already targets the **same physical phenomena** at a high level, but through a **different numerical stack**.

---

## Slide-by-theme analysis

### 1. Governing equations (momentum, energy, incompressibility)

**Slides show:**

- \( \rho(\partial_t \mathbf{U} + \mathbf{U}\cdot\nabla\mathbf{U}) = \nabla\cdot\sigma + \mathbf{F} \)
- \( \nabla\cdot\mathbf{U} = 0 \)
- Energy equation with advection, conduction, and volumetric heat source \( \dot{Q} \)

**Good for us:**

- Confirms our force budget is directionally correct: **Marangoni, Lorentz, buoyancy, arc heating** are the right ingredients.
- Supports documenting `waam_twin` against standard weld-pool theory (see `docs/weld_pool_physics.md`).

**Bad / mismatched:**

- Our momentum equation is solved via **LBM**, not direct NS discretization.
- We do not assemble global `[K]{U}={F}` systems each timestep.

**Actionable takeaway:** use these slides in thesis/docs as **theoretical grounding**, not implementation pseudocode.

---

### 2. Boundary conditions (symmetry, S–L interface, flat top surface)

**Slides show:**

- **Symmetry plane:** \(u=0\), \( \partial v/\partial x = 0\), \( \partial w/\partial x = 0\), zero heat flux
- **Solid–liquid interface:** no-slip \(u=v=w=0\)
- **Top surface:** Marangoni shear
  \[
  \mu\frac{\partial u}{\partial z} = f_L \frac{\partial\gamma}{\partial T}\frac{\partial T}{\partial x},\quad
  \mu\frac{\partial v}{\partial z} = f_L \frac{\partial\gamma}{\partial T}\frac{\partial T}{\partial y},\quad
  w=0
  \]
- **Top/side cooling:** convection + radiation
- **Arc input:** Gaussian heat flux on top surface

**Good for us:**

- Excellent **validation checklist** for whether our boundary physics are plausible.
- Marangoni BC form matches what we implement via **CSF + surface tension gradient** in `kernels.py` / `physics/forces.py`.
- Confirms that neglecting plasma drag on the pool surface (as the slides state) is a standard assumption when Marangoni dominates.

**Bad / mismatched:**

- Slides assume a **flat** top surface; we use **VOF** and can represent deformed surfaces and bead crown.
- Our symmetry is usually encoded in **domain placement / path**, not as an explicit symmetry BC plane in the slides' FEM sense.
- S–L interface in our model is **enthalpy/porosity + flags**, not a meshed interface with classical no-slip FEM constraints.

**Actionable takeaway:**

- Add a validation note: for symmetric bead-on-plate runs, compare against flat-surface Marangoni-outward expectations from Kou-type theory.
- Do **not** flatten our VOF surface just to match the slides.

---

### 3. Quasi-steady / moving-frame heat transfer

**Slides show:**

- Energy equation in a quasi-steady moving reference frame
- Matrices `[H]`, `[C]`, `[S]`, `[S̄]` for conduction, advection, and heat capacity
- Global reduced form `[S]{Ṫ} + [H̄̄]{T} = {f}`

**Good for us:**

- Useful for **sanity-checking thermal trends**: travel speed, arc power, and pool length scaling.
- Helps explain why very long `N_STEPS` are needed in our transient model to approximate a developed pool.

**Bad / mismatched:**

- Our solver is **fully transient**, not quasi-steady FEM.
- We do not solve a global thermal matrix each step; we use explicit/grid-based thermal updates.
- WAAM adds **deposition, layer buildup, and path corners** that break pure quasi-steady assumptions.

**Actionable takeaway:**

- Use quasi-steady theory only for **order-of-magnitude estimates**, not as a target architecture.
- Example: estimate steps for one path lap from \( \text{path length} / (v \Delta t) \).

---

### 4. Finite element discretization (8-node bricks, Galerkin, penalty method)

**Slides show:**

- 8-node isoparametric brick elements
- Shape-function expansions for \(u,v,w,T\)
- Penalty method for continuity: \(P = -\lambda(\nabla\cdot\mathbf{U})\)
- Reduced integration for penalty term
- Global momentum system with `[M]`, `[C]`, `[K]`, `[K̂]`

**Good for us:**

- Explains why classical weld CFD codes are expensive: global matrix assembly + implicit solves.
- Clarifies why our **LBM + explicit** approach is a legitimate alternative, not a "missing FEM implementation."

**Bad / mismatched:**

- **Not useful to port directly.** This would mean replacing the entire solver.
- Penalty incompressibility and LBM incompressibility are different mechanisms.
- 8-node unstructured meshes are irrelevant to our structured grid.

**Actionable takeaway:** **Do not** start a FEM branch unless the project scope explicitly changes to "build a DebRoy-style FEM solver."

---

### 5. Time discretization (implicit effective stiffness)

**Slides show:**

- Implicit time stepping with effective matrices like
  \[
  [\bar{K}] = \tfrac{2}{3}[\bar{K}] + \tfrac{1}{\Delta t}[M]
  \]
- Final solve `[K̄̄]{U²} = {F'}`

**Good for us:**

- Reinforces that **Δt is a core stability/accuracy knob** — same lesson applies to LBM.
- Useful conceptual parallel when explaining why `dx` and `dt` are linked in our grid setup.

**Bad / mismatched:**

- We do not use implicit FEM timestepping.
- Our export/runtime cost is dominated by **per-step explicit physics + VTK**, not sparse matrix factorization.

**Actionable takeaway:** document our timestep policy in `platform.py` / `grid.py` as the analogue of their time-discretization slide.

---

### 6. Electromagnetic force analytical expressions

**Slides show:**

- Closed-form \(F_{em}^x, F_{em}^y, F_{em}^z\) as functions of current, effective radius, and geometry
- Intended as body-force source terms in the momentum solve

**Good for us:**

- Strong validation reference for our **optional Lorentz** path (`enable_lorentz`, `physics/lorentz_physics.py`).
- Useful for checking order-of-magnitude of electromagnetic pumping vs Marangoni.
- Could inspire a **benchmark case**: compare our Lorentz acceleration scale against analytical estimate at fixed current.

**Bad / mismatched:**

- Our Lorentz implementation is a **numerical J×B model on the grid**, not necessarily these exact analytical formulae.
- Slides may omit WAAM-specific EM–droplet coupling near the wire tip.

**Actionable takeaway:**

- Add a validation test: at fixed `current_A`, compare peak Lorentz body force magnitude to slide-based estimate within an order of magnitude.
- Good thesis material: "comparison to classical EM weld-pool theory."

---

### 7. Surface tension coefficient vs temperature and solute activity

**Slides show:**

- \( \partial\sigma/\partial T \) model with solute activity terms (Sahoo/DebRoy-type formulation)
- Explains sign change of Marangoni direction with surface-active elements

**Good for us:**

- Directly relevant to **calibration** and `materials/` tables for `dγ/dT`.
- Explains why ER70S-6 behavior may differ from textbook "outward Marangoni" pure-metal cases.
- Supports interpreting inverted pool shapes or unexpected bead width in results.

**Bad / mismatched:**

- We may not model full solute activity chemistry explicitly.
- Our `dγ/dT` may come from material YAML tables rather than the full thermodynamic activity expansion.

**Actionable takeaway:**

- Map slide variables to our material schema: `sigma_liquid`, `dgamma_dT`, surfactant/activity flags if present.
- Use this slide to justify **why calibration exists** in the project.

---

### 8. Conservation of energy in full domain vs momentum only in pool

**Slides show:**

- Energy equation solved in **entire domain** (solid + liquid)
- Mass/momentum only in **liquid weld pool** region

**Good for us:**

- This matches our architecture closely:
  - thermal field on full grid
  - fluid/deposition logic active only in liquid / near-surface regions
- Good documentation alignment for thesis chapter on "domain partitioning."

**Bad / mismatched:**

- Our partitioning is via **flags/VOF/enthalpy**, not explicit FEM subdomains.

**Actionable takeaway:** cite this as evidence our domain split is physically standard.

---

### 9. FEM implementation workflow (mesh, reorder, banded solver, frontal solver)

**Slides show:**

1. Geometry → node/element list
2. Reorder/renumber for bandwidth
3. Estimate bandwidth/frontwidth
4. Assemble global matrices
5. Solve with banded, frontal, or iterative (LIS) methods

**Good for us:**

- Explains how classical weld FEM codes are structured.
- Useful if comparing runtime/complexity against our approach in a thesis "related work" section.

**Bad / mismatched:**

- **Zero direct implementation value** for `waam_twin`.
- Our workflow is: preset → `auto_grid` → `run_path` → VTK export.

**Actionable takeaway:** do not import any of this into the codebase.

---

## Crosswalk: slide concepts → `waam_twin` features

| Slide concept | Present in `waam_twin`? | Notes |
|---------------|-------------------------|-------|
| Energy equation in full domain | Yes | Enthalpy/temperature on full grid |
| Momentum in liquid pool only | Yes | LBM + flags/VOF |
| Marangoni convection | Yes | CSF, `dγ/dT` tables |
| Lorentz electromagnetic force | Partial | Optional; not identical analytical form |
| Buoyancy | Yes | Boussinesq-style body force |
| Recoil / keyhole pressure | Partial | Optional flags |
| Gaussian/Goldak heat input | Yes | Goldak in jobs |
| Convective/radiative losses | Yes | `heat_loss` in job YAML |
| Flat free surface | No (by design) | We use VOF |
| Symmetry BC | Partial | Problem setup, not explicit FEM BC |
| Quasi-steady formulation | No | Transient explicit |
| 8-node brick FEM | No | Structured grid |
| Penalty incompressibility | No | LBM |
| Droplet/wire deposition | Yes | Beyond classic slides |
| Bead geometry emergence | Yes | Core project goal |
| Global sparse FEM solve | No | Not applicable |

---

## What is genuinely useful for the project

### A. Validation and thesis documentation

Use the slides to define **expected physics behavior**:

- outward vs inward Marangoni trends
- Lorentz deepening at higher current
- role of surface-active elements
- symmetric bead-on-plate assumptions
- convective/radiative loss effects on tail temperature

These strengthen the **scientific credibility** of `waam_twin` without changing code.

### B. Calibration targets

Slides suggest what to tune against:

| Observable | Slide source | Our telemetry / export |
|------------|--------------|------------------------|
| Pool width | BC + Marangoni | `pool_width_mm` |
| Pool depth | Lorentz + flow | `pool_depth_mm` |
| Surface temperature gradient | Marangoni BC | `Temperature_K` VTK |
| HAZ extent | Energy equation | `T_max_K`, probes |
| Bead crown shape | Surface flow | surface VTP, `bead_height_mm` |

### C. Missing or weak areas to consider

| Slide emphasis | Our gap | Priority |
|----------------|---------|----------|
| Analytical EM benchmark | Lorentz validation is coarse | Medium |
| Solute/activity effect on σ(T) | May be simplified in materials | Medium |
| Quasi-steady reference solutions | No fast 1D/2D reference mode | Low |
| Symmetry benchmark case | Not packaged as standard job | Low |

### D. Good "related work" narrative

For reports/thesis:

> Classical weld-pool models use coupled FEM of Navier–Stokes and energy equations with Marangoni and Lorentz forces. Our digital twin preserves the same dominant physics but replaces global FEM with GPU LBM + VOF for transient WAAM path simulation, deposition, and VTK-based validation.

That is a strong story.

---

## What would be bad to adopt

### 1. Rewriting the solver as FEM

**Cost:** months to years  
**Benefit:** marginal for current FYP goal  
**Reason:** would discard Taichi LBM, VOF, deposition, cloud workflow, and existing tests

### 2. Imposing flat free-surface assumption

Would **reduce** fidelity for WAAM bead crown and deposition geometry — opposite of project goals.

### 3. Switching to quasi-steady thermal mode

Would break:

- path transients
- corner behavior on `bead_line.csv`
- layer deposition timing
- sequence exports over time

### 4. Implementing banded/frontal solvers

Irrelevant to explicit LBM time marching.

### 5. Treating slides as proof the current model is "wrong"

The slides describe a **different valid modeling tradition**, not a refutation of LBM/VOF.

---

## Recommended use of the slides in the project

### Short term (no major code changes)

1. **Add a "Classical FEM reference" subsection** to thesis/docs citing these concepts.
2. **Create one symmetry benchmark job** (half-domain bead-on-plate) and compare pool W/D trends to slide expectations.
3. **Add Lorentz order-of-magnitude test** against analytical EM slide formulae.
4. **Document Marangoni sign** for ER70S-6 from material `dγ/dT` and slide theory.

### Medium term (targeted improvements)

1. Improve `materials/` metadata for `dγ/dT` and surface-active behavior.
2. Add a "validation mode" notebook cell that prints force ratios: Marangoni vs Lorentz vs buoyancy.
3. Package a ParaView recipe aligned with slide diagnostics (surface temperature gradient, pool isotherms).

### Long term (only if scope expands)

1. Separate **reference FEM benchmark** (external code or reduced model), not in-repo replacement.
2. Optional **1D/2D quasi-steady Rosenthal/Goldak** analytical comparison tool for fast parameter sweeps.

---

## Suggested reading map (slides → repo files)

| Slide topic | Read next in repo |
|-------------|-------------------|
| Marangoni / surface tension | `docs/weld_pool_physics.md`, `kernels.py` CSF section |
| Lorentz force | `physics/lorentz_physics.py`, `validation/test_lorentz_physical_scale.py` |
| Energy / solidification | `physics/thermal.py`, `physics/phase_change.py` |
| Boundary / wetting | `physics/free_surface.py`, job `surface_wetting` |
| Arc heat source | `kernels.py` Goldak, `jobs/examples/bead_on_plate.yaml` |
| Time step / grid | `platform.py`, `grid.py` |
| Export / validation | `export/bundle.py`, `validation/` |

---

## Final recommendation

**Yes, the slides are useful — but as physics and validation reference material, not as an implementation spec.**

They are especially valuable for:

- explaining *why* the project includes Marangoni, Lorentz, buoyancy, and heat-loss terms
- defining *what* pool behavior should look like in limit cases
- writing thesis "background" and "model assumptions" sections
- identifying a few **targeted validation gaps** (Lorentz scale, σ(T) activity effects)

They are **not** a signal to pivot `waam_twin` toward classical FEM. The current LBM + VOF + deposition architecture is addressing a **richer WAAM problem** than the slide set assumes.

**Practical stance:**

- **Adopt the physics vocabulary**
- **Reject the numerical infrastructure**
- **Borrow validation cases selectively**

---

## Appendix: image inventory (themes)

The uploaded slide set covers, among others:

- Governing equations and boundary conditions (transport)
- Conservation of energy
- Boundary conditions (velocity / symmetry / flat surface)
- Finite element discretization (Galerkin, shape functions)
- Penalty method for continuity
- Heat transfer quasi-steady formulation
- Time discretization / effective stiffness
- Electromagnetic force analytical expressions
- Surface tension vs temperature and solute activity
- Solution of energy equation (FEM assembly)
- Implementation of FEM (mesh, bandwidth, solvers)
- Final coupled matrix equations for energy and momentum

Together these form a coherent **FEM weld-pool course**, not a WAAM digital-twin codebase — but a strong **reference layer** for this project.
