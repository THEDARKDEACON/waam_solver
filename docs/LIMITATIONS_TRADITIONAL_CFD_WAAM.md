# Limitations of Traditional CFD Tools for WAAM: Computational Bottlenecks and Dynamic Free-Surface Tracking

## Abstract

This document evaluates principal limitations of conventional computational fluid dynamics (CFD) platforms—principally Ansys Fluent / Mechanical-class finite-volume solvers and OpenFOAM-style open-source CFD—when applied to wire-arc additive manufacturing (WAAM) melt-pool modelling. Two themes are developed in depth: (i) **computational bottlenecks** arising from mesh refinement, Courant–Friedrichs–Lewy (CFL) time-step restrictions, and tightly coupled nonlinear solves; and (ii) **dynamic free-surface tracking**, typically realised through volume-of-fluid (VOF) or related interface-capturing methods. The discussion is intended for formal academic submission. It does not dismiss industrial CFD; rather, it situates such tools as high-fidelity reference instruments whose cost and numerical stiffness constrain their use as the sole engine for large calibration and prediction campaigns. Within the present project, Ansys remains a designated referee (thermal Tier A; Fluent VOF Tier B), while iterative twin campaigns are executed in `waam_twin`.

---

## 1. Introduction and problem statement

Wire-arc additive manufacturing deposits successive weld beads by robotic gas metal arc welding (GMAW / MIG). Beneath the torch, a liquid metal pool forms; behind the torch the metal solidifies into a bead characterised by crown height, fusion width, and penetration depth. Subsequent passes remelt portions of prior layers. Predictive simulation of this process must therefore resolve coupled heat transfer, melting and solidification, free-surface evolution under surface tension (including thermocapillary Marangoni flow), arc pressure, and, where enabled, electromagnetic (Lorentz) forcing, together with continuous mass addition from the consumable wire.

![GMAW weld area diagram](figures/gmaw_weld_area.png)

**Figure 1.** Schematic of the GMAW weld region, the process family underlying WAAM. Key: (1) direction of travel; (2) contact tube / electrode guide; (3) electrode wire; (4) shielding gas; (5) molten weld metal; (6) solidified weld metal; (7) workpiece. Source: Wikimedia Commons, *[GMAW weld area.png](https://commons.wikimedia.org/wiki/File:GMAW_weld_area.png)* (CC BY-SA 3.0), after conventional Lincoln Electric Procedure Handbook diagramming practice.

Quantities of engineering interest include fusion-zone width and depth as functions of current, voltage, travel speed and wire-feed rate; the response of bead reinforcement to wire feed; and the stability of those predictions under mesh refinement. Although these questions are well posed, the governing physics is stiff: intense localised heating, a moving multiphase interface, and deposition-driven free-surface growth over seconds of physical process time.

Traditional CFD packages solve the continuum conservation equations for mass, momentum and energy on meshes of control volumes, using segregated or coupled pressure–velocity algorithms (e.g. SIMPLE, PISO, or fully coupled formulations). This methodology is mature for many industrial flows. WAAM, however, stresses two regimes in which continuum CFD is characteristically expensive: the requirement for **fine spatial discretisation with correspondingly restrictive temporal resolution**, and the need for **accurate, sharp tracking of a dynamically evolving free surface**.

---

## 2. Computational bottlenecks

### 2.1 Spatial discretisation and mesh cost

Finite-volume CFD partitions the computational domain into discrete cells in which field variables (velocity, pressure, temperature, and, for multiphase models, phase fraction) are stored. Coarse meshes under-resolve fusion geometry and free-surface curvature; meshes fine enough for millimetre-scale fusion metrics imply large cell counts. In three dimensions, cell count scales approximately as \(1/\mathrm{d}x^{3}\). Halving the characteristic cell size therefore increases the number of cells by a factor of approximately eight. Transition from a shop-scale spacing of order \(0.4\,\mathrm{mm}\) to \(0.2\,\mathrm{mm}\) consequently constitutes a change of computational class, not a minor parametric adjustment.

![Structured CFD grid](figures/structured_grid.png)

**Figure 2.** Example of a structured CFD grid. Source: Wikimedia Commons, *[Structured Grid.PNG](https://commons.wikimedia.org/wiki/File:Structured_Grid.PNG)* (CC BY-SA).

![Unstructured CFD grid](figures/unstructured_grid.jpg)

**Figure 3.** Example of an unstructured CFD grid. Unstructured meshes accommodate complex geometry but complicate local refinement about a moving torch. Source: Wikimedia Commons, *[Unstructured Grid.jpg](https://commons.wikimedia.org/wiki/File:Unstructured_Grid.jpg)* (CC BY-SA 3.0).

A WAAM-specific difficulty is that the region of interest—the melt pool and growing crown—translates with the torch. Uniform fine meshing of the entire plate allocates the majority of degrees of freedom to thermally inactive material. Dynamic remeshing, overset methods, or adaptive mesh refinement can concentrate resolution near the torch, but increase setup complexity and parallel load-balancing cost. Commercial and open-source solvers provide such capabilities; none removes the underlying scaling of work with resolution.

### 2.2 Temporal discretisation and CFL constraints

Transient CFD advances the solution in discrete time increments \(\Delta t\). Admissible step size is constrained by accuracy and, for many free-surface schemes, by stability related to the Courant–Friedrichs–Lewy (CFL) condition. A representative Courant number is

\[
C \approx u\,\frac{\Delta t}{\Delta x},
\]

where \(u\) is a characteristic velocity and \(\Delta x\) a local cell size. Elevated surface velocities associated with Marangoni flow, combined with fine \(\Delta x\) near the interface, force small \(\Delta t\) if \(C\) is to remain of order unity or below—particularly for multiphase phase-fraction updates. Industrial guidance for transient multiphase analyses commonly recommends keeping interfacial Courant numbers of order one or less [SimScale knowledge base](https://www.simscale.com/knowledge-base/what-is-a-courant-number/).

![Effect of Courant number on a simple advection scheme](figures/courant_stability.png)

**Figure 4.** Illustration of the influence of Courant number \(C\) on a first-order upwind discretisation of one-dimensional advection: excessive \(C\) produces unstable behaviour. Production multiphase solvers employ more advanced schemes, yet the underlying pressure on \(\Delta t\) remains. Source: Wikimedia Commons, *[First Order Upwind Scheme Stability.png](https://commons.wikimedia.org/wiki/File:First_Order_Upwind_Scheme_Stability.png)* (CC BY-SA 4.0).

Consequently, although torch travel speeds are typically of order \(10\,\mathrm{mm\,s^{-1}}\), numerical time steps may fall in the microsecond range. Several seconds of welding then require tens of thousands to millions of steps—the primary temporal bottleneck.

### 2.3 Coupled nonlinear solution procedures

Each time step generally comprises iterative solution of several tightly coupled subsystems, including:

1. **Momentum and pressure** — enforcement of (approximate) continuity through pressure–velocity coupling (SIMPLE, PISO, or coupled solvers), involving repeated linear algebra;
2. **Energy and phase change** — temperature and liquid-fraction updates, with strong damping in the mushy zone;
3. **Free-surface / VOF transport** — evolution of phase fraction and interface geometry;
4. **Optional multiphysics** — Lorentz body forces, arc pressure, evaporative recoil, turbulence closures, or species transport.

Outer iterations dominate wall-clock time on large meshes. Stiff Marangoni or vapour-recoil forcing may necessitate further reduction of \(\Delta t\), increased under-relaxation, or selective deactivation of physical mechanisms to obtain convergence. The latter practice risks producing a converged solution that no longer represents the intended process physics.

### 2.4 Campaign cost versus single-case demonstration

A single high-quality Fluent or OpenFOAM demonstration case may justify substantial engineer effort for publication or design review. Digital-twin development, by contrast, requires **campaigns**: calibration, held-out prediction without retuning, mesh-convergence studies, and process-parameter sweeps. Commercial licensing (Ansys), high-performance computing queue time, mesh generation, and user-defined extensions (Fluent UDFs; OpenFOAM `fvOptions` and related infrastructure) multiply that cost. OpenFOAM eliminates licence fees but does not eliminate human or HPC expenditure.

The central computational limitation is therefore less a question of whether continuum CFD *can* represent a weld pool than whether it can be exercised **repeatedly** at research resolution during model development. Published WAAM CFD efforts report correspondingly large resource requirements (small \(\Delta t\), multiple transport equations, substantial core-hour budgets); see, for example, the Linköping SNIC/OpenFOAM melt-pool project summary ([NIM project 4200](https://nim.nsc.liu.se/projects/4200/)) and the review *Numerical modelling of wire arc additive manufacturing* ([DOI 10.1088/1361-6463/adb3b0](https://doi.org/10.1088/1361-6463/adb3b0)).

---

## 3. Dynamic free-surface tracking

### 3.1 Definition and engineering significance

The bead crown constitutes a free surface separating liquid metal, shielding gas (or air), and, in the wake, solidified deposit. The interface rises under wire deposition, spreads under wetting and surface tension, may oscillate under droplet impact, and must remain consistent with the solid substrate at the toe. Errors in free-surface prediction degrade bead height and toe geometry even when an internal fusion envelope appears plausible.

### 3.2 Volume-of-fluid methodology

Industrial practice predominantly employs the **volume-of-fluid (VOF)** method. Rather than storing an explicit geometric skin, each cell carries a phase fraction \(\alpha\) (or equivalent) with

- \(\alpha = 0\): cell empty of the tracked liquid phase,
- \(\alpha = 1\): cell filled with liquid,
- \(0 < \alpha < 1\): interface cell.

A standard overview is given by Wikipedia’s *[Volume of fluid method](https://en.wikipedia.org/wiki/Volume_of_fluid_method)* article and the foundational literature cited therein (e.g. Hirt and Nichols, 1981). Phase fraction records the *quantity* of liquid in a cell but not, by itself, the *geometry* of the cut. Solvers therefore **reconstruct** an interface—commonly via piecewise linear interface calculation (PLIC)—and advect liquid with geometric or algebraic flux schemes designed to preserve boundedness and sharpness.

![VOF volume-fraction representation](figures/vof_scheme.png)

**Figure 5.** Schematic of VOF cells: filled, empty, and partially filled interface cells. Source: Wikimedia Commons, *[Vof scheme.png](https://commons.wikimedia.org/wiki/File:Vof_scheme.png)*.

![VOF with PLIC reconstruction](figures/vof_scheme_plic.png)

**Figure 6.** VOF representation with piecewise-linear (PLIC) interface segments within interface cells. Source: Wikimedia Commons, *[Vof scheme with plic.png](https://commons.wikimedia.org/wiki/File:Vof_scheme_with_plic.png)*.

![PLIC reconstruction of a droplet](figures/plic_vof_droplet.png)

**Figure 7.** Example droplet represented by PLIC reconstruction in a VOF computation; planar cuts per cell yield a faceted appearance in under-resolved regions. Source: Wikimedia Commons, *[Droplet - PLIC VOF.png](https://commons.wikimedia.org/wiki/File:Droplet_-_PLIC_VOF.png)*.

![VOF free-surface simulation illustration](figures/vof_drifter_drop.png)

**Figure 8.** Illustration of free-surface fluid simulation using a VOF-type method. Source: Wikimedia Commons, *[Drifter drop.png](https://commons.wikimedia.org/wiki/File:Drifter_drop.png)* (also reproduced on the English Wikipedia VOF article).

OpenFOAM documentation describes bounded VoF transport (e.g. MULES) and interface-compression / PLIC options intended to maintain sharp interfaces ([CFD Direct — Interface capturing in OpenFOAM](https://cfd.direct/openfoam/free-software/multiphase-interface-capturing/)). Ansys Fluent provides a comparable VOF multiphase capability with analogous objectives and trade-offs.

### 3.3 Limitations of VOF for WAAM free surfaces

The following issues do not invalidate VOF; they characterise the stiffness of **VOF combined with welding multiphysics**.

1. **Interface smearing.** Numerical diffusion can broaden a sharp crown into bands of intermediate phase fraction. Compression schemes counteract smearing but may introduce additional artefacts.

2. **Surface-tension evaluation.** Continuum surface force formulations require accurate interface normals and curvature. Noisy reconstruction produces **parasitic currents**—spurious near-interface velocities—that can distort crown height and toe angle.

3. **Coupling to temporal restrictions.** Strong Marangoni surface flow elevates local Courant numbers and reinforces the \(\Delta t\) bottleneck identified in Section 2.

4. **Mass addition at the free surface.** Continuous wire deposition (discrete droplets or equivalent volumetric sources) requires the free surface to accommodate added liquid volume. Inadequate geometric capacity or inconsistent placement yields voids, overflows, or imbalance between deposited mass and wire feed. Related mass-ledger failures observed on fine meshes in the project twin illustrate the same discrete-interface difficulty under different numerics.

5. **Wetting and contact-line behaviour.** Static contact-angle models are widely used at the liquid–solid toe, yet predicted geometry can vary with mesh resolution.

6. **Coupling to solidification.** Downstream of the torch, metal solidifies while neighbouring liquid continues to flow. Numerical freezing of the crown without corrupting the established free-surface geometry adds further coupling complexity.

Achieving a stable, mesh-converged WAAM crown in Fluent or OpenFOAM is therefore frequently a research contribution in its own right. Published OpenFOAM CMT–WAAM melt-pool studies (e.g. in the *Journal of Advanced Research in Numerical Heat Transfer*) demonstrate feasibility, but typically under specialised modelling assumptions and substantial computational cost.

### 3.4 Level-set and hybrid formulations

Level-set methods track a signed-distance function to the interface and can furnish smoother normals, benefiting surface-tension evaluation, but are susceptible to mass drift unless corrected. Hybrid VOF–level-set schemes seek to combine advantages of both approaches. Neither removes the fundamental requirements of fine interfacial resolution and careful temporal discretisation.

---

## 4. Synthesis of limitations

| Limitation | Formal implication | Relevance to WAAM |
|------------|--------------------|-------------------|
| Mesh cost \(\propto 1/\mathrm{d}x^{3}\) | Refinement multiplies degrees of freedom rapidly | Fusion W/D and crown curvature demand fine cells |
| CFL / Courant restrictions | Surface velocity limits \(\Delta t\) | Marangoni + VOF near the crown |
| Coupled nonlinear solvers | High cost per step | Pressure–velocity, energy, VOF, optional EM |
| Free-surface reconstruction | Interface geometry is approximate | Bead height and toe sensitive to artefacts |
| Deposition with interface tracking | Mass addition and \(\alpha\)-transport interact | Continuous wire feed |
| Workflow, licensing, and HPC operations | Human and queue time dominate campaigns | Twin development requires many runs |

The limitation of traditional CFD for WAAM is therefore not an inability to formulate a melt-pool problem, but the **resource intensity and numerical fragility** of full-physics free-surface solutions when employed as the primary engine for iterative digital-twin development. Predictions may remain mesh-dependent; free-surface setups remain delicate; and wall-clock cost limits the number of calibration and validation experiments that can be performed.

---

## 5. Implications for the present project

Consistent with the project methodology, responsibilities are partitioned as follows:

| Role | Tooling | Rationale |
|------|---------|-----------|
| Calibration, held-out prediction, mesh-band studies | `waam_twin` (lattice Boltzmann with free-surface options) | Campaign throughput on GPU hardware |
| Thermal fusion cross-check | Ansys Tier A | Independent referee |
| Melt-pool / VOF cross-check | Fluent Tier B (as available) | Independent free-surface referee |
| Live cell supervision | Distilled soft sensor | Full CFD/LBM is not suitable for PLC-rate control without distillation |

Related project documentation:

- [FYP_METHODOLOGY_GROUNDED.md](FYP_METHODOLOGY_GROUNDED.md)  
- [validation/ansys_2024r2/SETUP_FLUENT.md](validation/ansys_2024r2/SETUP_FLUENT.md)  
- [report/05 - Free Surface and Wetting.md](report/05%20-%20Free%20Surface%20and%20Wetting.md)  
- [physics/LBM.md](physics/LBM.md)

---

## 6. Conclusions

1. **Computational bottlenecks.** Application of Ansys- and OpenFOAM-class CFD to WAAM is constrained by three-dimensional mesh refinement costs, CFL-limited time steps associated with interfacial flow, and expensive coupled pressure–velocity, energy, and multiphase iterations at each step. Physical welding times of seconds can require wall-clock times of hours to days at research resolution.

2. **Dynamic free-surface tracking.** VOF and related methods can represent a growing bead, but interface reconstruction quality, surface-tension artefacts, wetting, solidification coupling, and continuous deposition render free-surface tracking one of the most demanding components of the model—often more fragile than thermal source calibration alone.

3. **Recommended practice.** Traditional CFD is appropriately employed as a **high-fidelity referee** and for specialised studies. A complementary twin architecture is appropriate for calibration campaigns, held-out prediction, and mesh-dependence assessment. Neither full-physics CFD nor a full twin should be assumed suitable for real-time process control without model reduction or soft-sensor distillation.

---

## Figure sources and licensing

Local copies are stored under `docs/figures/`. Licence text should be verified on the corresponding Wikimedia Commons file page prior to formal publication.

| File | Source | Licence (indicative) |
|------|--------|----------------------|
| `gmaw_weld_area.png` | [GMAW weld area.png](https://commons.wikimedia.org/wiki/File:GMAW_weld_area.png) | CC BY-SA 3.0 |
| `structured_grid.png` | [Structured Grid.PNG](https://commons.wikimedia.org/wiki/File:Structured_Grid.PNG) | CC BY-SA |
| `unstructured_grid.jpg` | [Unstructured Grid.jpg](https://commons.wikimedia.org/wiki/File:Unstructured_Grid.jpg) | CC BY-SA 3.0 |
| `courant_stability.png` | [First Order Upwind Scheme Stability.png](https://commons.wikimedia.org/wiki/File:First_Order_Upwind_Scheme_Stability.png) | CC BY-SA 4.0 |
| `vof_scheme.png` | [Vof scheme.png](https://commons.wikimedia.org/wiki/File:Vof_scheme.png) | see Commons |
| `vof_scheme_plic.png` | [Vof scheme with plic.png](https://commons.wikimedia.org/wiki/File:Vof_scheme_with_plic.png) | see Commons |
| `plic_vof_droplet.png` | [Droplet - PLIC VOF.png](https://commons.wikimedia.org/wiki/File:Droplet_-_PLIC_VOF.png) | see Commons |
| `vof_drifter_drop.png` | [Drifter drop.png](https://commons.wikimedia.org/wiki/File:Drifter_drop.png) | see Commons |

Copyrighted journal result plots from Fluent or OpenFOAM WAAM studies are not reproduced here; they are cited by DOI or publisher URL.

---

## References

1. Hirt, C. W. and Nichols, B. D. (1981). Volume of fluid (VOF) method for the dynamics of free boundaries. *Journal of Computational Physics*, 39(1), 201–225.  
2. Wikipedia contributors. *Volume of fluid method*. https://en.wikipedia.org/wiki/Volume_of_fluid_method  
3. CFD Direct. *Interface capturing in OpenFOAM*. https://cfd.direct/openfoam/free-software/multiphase-interface-capturing/  
4. SimScale. *What is a Courant Number?* https://www.simscale.com/knowledge-base/what-is-a-courant-number/  
5. *Numerical modelling of wire arc additive manufacturing: methods, status, trends, and opportunities*. DOI: [10.1088/1361-6463/adb3b0](https://doi.org/10.1088/1361-6463/adb3b0)  
6. National Supercomputer Centre / Linköping University. CFD simulation of metal transfer and melt pool in additive manufacturing with electric arc (project summary). https://nim.nsc.liu.se/projects/4200/  
7. OpenFOAM CMT-WAAM melt-pool modelling example: *Journal of Advanced Research in Numerical Heat Transfer*, https://semarakilmu.com.my/journals/index.php/arnht/article/view/13806  
)
