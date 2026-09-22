---
title: What I Gathered From the Papers
step: 8
tags:
  - report/step
  - literature
  - bibliography
up: "[[WAAM Digital Twin — Project Report]]"
prev: "[[07 - Deposition and Metal Transfer]]"
next: "[[09 - The Physics Force Correctness Campaign]]"
---

# 08 — What I Gathered From the Papers

> [!summary] The step in one sentence
> Every equation in the twin is traceable to a source, and I maintained that traceability in two places at once — a LaTeX equation catalogue with 46 bibliography entries, and attribution comments in the kernels themselves.

---

## How I organised the reading

I kept two parallel documents alongside the code. `docs/WAAM_FORMULAE.tex` is my equation catalogue: every formula the twin implements, with a citation attached to each one. `docs/WAAM_WELD_POOL_PHYSICS_CENTRE.md` is the prose companion, organised by physical mechanism, with a master reference list.

> [!tip] The habit that saved me most often
> I put the citation **in the kernel comment**, next to the code. When I later found the Lorentz magnitude was wrong, the comment told me which paper's formulation I had thought I was implementing, which made the discrepancy obvious. A bibliography at the end of a thesis cannot do that.

Two reviews carried most of my orientation work. **Aryal (2023)**, a PhD thesis from University West on pulsed-GMAW melt-pool modelling and CFD, is my single most-cited source — it gave me the modern operator ordering for a GMAW CFD stack, the droplet detachment force diagram, and a comparative view of electromagnetic force formulations. **Love et al. (2025)**, *Advancing Metal Additive Manufacturing: A Review of Numerical Methods in DED, WAAM, and PBF*, gave me the interlayer and multi-layer framing.

---

## Heat source

| Source | What I took |
|---|---|
| **Goldak, Chakravarti & Bibby (1984)** — *A new finite element model for computing heat flow in welds*, Welding J. 63(10) | The double-ellipsoid volumetric source, front/rear fractions summing to two, and the axis naming convention I follow |
| **Goldak & Akhlaghi (1984)** — *Computational weld pool geometry* | The geometry-side companion: how pool width, depth and crown are defined for comparison |
| **Rykalin (1951)** — *Berechnung der Wärmevorgänge beim Schweißen* | The classical Gaussian surface flux, provenance for my cheap `Gaussian2D` model |
| **Kazemi & Goldak (2009)** | Supporting context for the double-ellipsoid family; not implemented |

See [[04 - The Arc Heat Source]].

---

## Thermal transport and phase change

| Source | What I took |
|---|---|
| **Rosenthal (1946)** — *The theory of moving sources of heat* | The analytical moving-source solution, implemented purely as a validation target |
| **Carslaw & Jaeger (1959)** — *Conduction of Heat in Solids* | Fourier conduction and the Stefan benchmark formulation |
| **Crank (1984)** — *Free and Moving Boundary Problems* | Stefan problem framing |
| **Voller & Prakash (1987)** — enthalpy–porosity for mushy-region phase change | The core of my thermal solver: enthalpy as primary variable, liquid-fraction recovery, Carman–Kozeny drag |
| **Brent, Voller & Reid (1988)** | The same formulation applied to convection–diffusion melting |
| **Poirier (1987)** | Permeability provenance for the Darcy constant |
| **Patankar (1980)** — *Numerical Heat Transfer and Fluid Flow* | Upwind advection with central diffusion |
| **Incropera et al. (2007)** | Convective and radiative boundary loss forms, property orders of magnitude |
| **Batchelor (1967)** | Bond number and the dimensionless-group reasoning about gravity versus surface tension |

See [[03 - Thermal Transport and Solidification]].

---

## Lattice Boltzmann numerics

| Source | What I took |
|---|---|
| **Qian, d'Humières & Lallemand (1992)** | The D3Q19 BGK equilibrium |
| **Guo, Zheng & Shi (2002)** — *Discrete lattice effects on the forcing term* | The forcing scheme every single body force passes through |
| **Chen & Doolen (1998)** | Lattice units and equilibrium conventions |
| **Krüger et al. (2017)** — *The Lattice Boltzmann Method: Principles and Practice* | My working handbook: relaxation times, Mach targets, force scaling |
| **Geier et al. (2015)** — *The cumulant lattice Boltzmann equation in three dimensions* | The central-moment collision structure that my SymPy generator follows |

> [!warning] Where I deviate from Geier
> My generated kernel relaxes only the second-order stress moments at two independent rates and projects higher moments straight onto equilibrium. That is not a cumulant operator. I documented the deviation rather than borrowing the name's authority — see [[02 - The Lattice Boltzmann Core]].

---

## Free surface, surface tension and wetting

| Source | What I took |
|---|---|
| **Brackbill, Kothe & Zemach (1992)** — *A continuum method for modeling surface tension* | The continuum surface force, $\kappa = -\nabla\cdot\hat{\mathbf{n}}$, and the Laplace-pressure acceptance test |
| **Hirt & Nichols (1981)** | The volume-of-fluid phase field |
| **Rider & Kothe (1998)** | Volume-tracking reconstruction context |
| **Ding & Spelt (2008)** — *Wetting condition in multiphase lattice Boltzmann simulations* | The ghost-$\phi$ contact-angle boundary condition |
| **de Gennes, Brochard-Wyart & Quéré (2004)** — *Capillarity and Wetting Phenomena* | Young's law, capillary length, Bond number |

See [[05 - Free Surface and Wetting]].

---

## Marangoni and surface chemistry

| Source | What I took |
|---|---|
| **Davis (1987)** — *Thermocapillary flows* | The Marangoni force form |
| **Heiple & Roper (1982)** — *Mechanism of minor element effect on GTA fusion zone geometry* | The physical mechanism: trace sulphur flips the sign of $d\gamma/dT$ and inverts pool shape |
| **Sahoo, DebRoy & McNallan (1988)** | The full Fe–S surface tension isotherm and its analytical derivative, which I implemented directly |
| **Mills, Keene, Brooks & Shirvanian (1998)** — *Marangoni effects in welding* | The force taxonomy I organise my catalogue around, and $d\gamma/dT$ magnitudes |
| **Burgardt & Heiple (1986)** | Fluid flow in high- and low-sulphur GTA welds |
| **Havrylov** (thesis excerpt) | Marangoni scaling and surface-active element behaviour |

> [!note] This is the clearest paper-to-code path in the project
> Heiple & Roper explained *why* sulphur matters. Sahoo, DebRoy & McNallan gave me the closed-form isotherm. Mills et al. gave me the magnitudes to sanity-check against. And my FEM assessment flagged solute-dependent surface tension as a validation gap I had not closed. Four sources, one implemented model, one test asserting the sign inversion above 150 ppm.

---

## Arc mechanical effects

| Source | What I took |
|---|---|
| **Lin & Eagar (1986)** — *Pressures produced by gas tungsten arcs* | $p_0 = \mu_0I^2/(4\pi^2\sigma_p^2)$, my arc-pressure closure |
| **Lin & Eagar (1985)** | The caveat that arc pressure may be secondary at high current — cited for interpretation only |
| **Oreper & Szekely (1984)** — *Heat and fluid flow phenomena in weld pools* | The foundational coupling of Navier–Stokes with Maxwell's equations; my reference for the whole force set |
| **Kou & Sun (1985)** | The electromagnetic pool model and the enclosed-current magnetic field form |
| **Kou (2003)** — *Welding Metallurgy* | General weld-pool physics and property orders |
| **Jackson** — *Classical Electrodynamics* | The potential formulation $\nabla\cdot(\sigma\nabla\varphi)=0$ and Ampère relations |
| **Tanaka & Lowke (2007)** | Cited as an explicit **non-scope** boundary: full arc-plasma MHD is not what I built |

See [[06 - The Weld Pool Force Catalogue]].

---

## Vapour, recoil and evaporation

| Source | What I took |
|---|---|
| **Matsunawa & Semak (1997)** and **Semak & Matsunawa (1997)** | Recoil pressure as the mechanical basis of surface depression; the enthalpy/vaporisation ceiling |
| **Anisimov** (via Hertz–Knudsen) | The evaporative mass flux form |
| **Knight** | The accommodation coefficient $C_\text{acc}\approx0.54$ — which my own sweep disagrees with |

---

## Deposition and metal transfer

| Source | What I took |
|---|---|
| **Wang, Hu & Carlson (2003)** — *Modelling and analysis of metal transfer in GMAW* | Wire mass rate, droplet mass and volume, impact velocity, impact pressure, droplet entry enthalpy |
| **Lancaster (1986)** — *The Physics of Welding* | Transfer-mode taxonomy and droplet physics |
| **Bernhardi & Duffie (2005)** | Droplet impact pressure and the requirement that deposited mass conserve to order unity |
| **Klaus et al. (2022, 2023)** | Contact-tip-to-work distance sensing via short-circuit resistance |
| **Zhao et al. (2021)** | GMAW-WAAM heat transfer, flow and geometry |
| **Ogino et al. (2020)** | Stable deposited height in GMAW additive manufacturing |

---

## Digital twin architecture

This set shaped [[01 - Project Genesis and Architecture Pivots]] rather than any equation.

| Source | What I took |
|---|---|
| **Kim, Shao & Jo (2022)** | A digital-twin implementation architecture for WAAM based on ISO 23247 |
| **Kim et al. (2025)** | Architecture development for digital-twin-based wire-arc directed energy deposition |
| **ISO 23247-1 (2021)** | The role mapping and vocabulary I initially organised around, then demoted |
| **Skryhunets et al. (2026)** | Multi-layer WAAM structure modelling for digital-twin integration |
| **Mu (2024)** (PhD, Wollongong) | Digital twin of wire-arc additive manufacturing |
| **Knapp et al. (2017)** — *Building blocks for a digital twin of additive manufacturing*, Acta Materialia | The canonical statement of what a printing digital twin needs |
| **Grieves & Vickers (2017)** | Digital-twin definition and synchronisation framing |
| **Åström & Murray (2008)** — *Feedback Systems* | Bounded supervisory setpoint correction |

---

## Experimental data sources

This is separate from the modelling literature and it matters more for credibility. See [[10 - Validation, Calibration and Results]] for how each was used.

| Dataset | What I took from it | Status |
|---|---|---|
| **ER70S-6 bead-on-plate macrograph** | Pool width **7.0 mm**, depth **3.0 mm** — my single calibration lock | Primary anchor |
| **Bruno GMAW dataset** (Figshare, DOI 10.6084/m9.figshare.27325698) | 110 A, 19.7 V, 6.0 mm/s; laser-scanned surface bead **4.33 × 2.14 mm** | Soft gate — wire grade uncertified, and this is external bead width, not a cut fusion zone |
| **PIONEER M1 wall** (Zenodo 17608626) | Böhler Q G 3 (= ER70S-6), 132 A, 14.3 V; wall width **5.41 mm**, remelt depth **2.0 mm**, layer height **1.488 mm** over 32 layers | Multipass reference; M2 excluded (different wire) |
| **CMT profile dataset** (Recherche Data Gouv / ENSAM Metz) | 140 beads, ER100 wire on S235, ~270 A | **Deliberately not wired in** — wrong wire, CMT transfer, far higher heat input |
| **Ansys 2024 R2** comparison | Shared inputs: 131 A × 10.4 V, $\eta=0.72$ → 981 W, same Goldak axes | Scaffolded, never run |

> [!warning] Digitisation honesty
> For the PIONEER walls I recorded exactly how I measured: widths from macrograph JPEGs using the 5 mm scale bar, taking the largest metal run with the 12 mm plate excluded — and one specimen where the scale-bar detection failed, so I left its width null rather than guessing. Remelt depth came from the darker-etch band beneath the wall. I would rather have a documented null than an undocumented number.

---

## Names I looked for and did not use

For completeness, since a reader might expect them: I found no use of Chapman–Enskog expansions, De Rosis or Kupershtokh forcing variants, Shan–Chen multiphase, Latva-Kokko, Smagorinsky turbulence, or Traidia and Hu & Tsai. **Cho & Na (2021)** appears as a methodological attribution for my force-ablation ranking style but I never obtained a full bibliographic entry for it, which I note as a gap.

---

## Related

- `docs/WAAM_FORMULAE.tex` — the equation catalogue with all 46 entries
- `docs/WAAM_WELD_POOL_PHYSICS_CENTRE.md` — the prose physics reference
- [[12 - Limitations and Open Work]] — datasets still awaiting measurement
