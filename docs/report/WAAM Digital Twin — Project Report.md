---
title: WAAM Digital Twin — Project Report
author: Gareth Joel
project: waam_twin v2.0.0
date: 2026-08-31
type: MOC
tags:
  - waam
  - digital-twin
  - lattice-boltzmann
  - report/index
---

# WAAM Digital Twin — Project Report

> [!abstract] What I built
> I built **`waam_twin`**, a GPU-accelerated digital twin of the **wire-arc additive manufacturing (WAAM) melt pool**. It couples a **D3Q19 lattice Boltzmann** fluid solver, an **enthalpy–porosity** solidification model, and a **volume-of-fluid** free surface, all written in **Taichi** so the same source runs on CUDA, Vulkan or CPU.
>
> The central claim I wanted to defend is this: **bead width and penetration depth are outputs, not inputs.** I never draw the bead in. It emerges from arc heating, Marangoni flow, surface tension, gravity, electromagnetic forces, vapour recoil and droplet deposition, all fighting each other inside a coupled timestep.

This note is the index. Each numbered note below is a step in how the project actually unfolded, and each one records both the technique I used and where I got it from in the literature.

---

## How to read this report

I wrote it in the order I worked, not in the order a textbook would present it. If you only read three notes, read [[01 - Project Genesis and Architecture Pivots]] to understand why the project looks the way it does, [[09 - The Physics Force Correctness Campaign]] to see the hardest debugging I did, and [[12 - Limitations and Open Work]] because that is where I am most honest about what the twin cannot yet do.

### The narrative arc

| # | Note | What this step was about |
|---|---|---|
| 01 | [[01 - Project Genesis and Architecture Pivots]] | Three architecture rewrites, and rejecting the FEM route |
| 02 | [[02 - The Lattice Boltzmann Core]] | D3Q19, collision operators, lattice units, Guo forcing |
| 03 | [[03 - Thermal Transport and Solidification]] | Enthalpy–porosity, minmod advection, Carman–Kozeny drag |
| 04 | [[04 - The Arc Heat Source]] | Gaussian vs Goldak double-ellipsoid, energy renormalisation |
| 05 | [[05 - Free Surface and Wetting]] | VOF advection, Brackbill CSF, contact-angle boundary condition |
| 06 | [[06 - The Weld Pool Force Catalogue]] | Marangoni, Lorentz, Lin–Eagar arc pressure, recoil, gas shear |
| 07 | [[07 - Deposition and Metal Transfer]] | Wire feed, droplet scheduling, mass ledger |
| 08 | [[08 - What I Gathered From the Papers]] | The full literature trail and what each source gave me |
| 09 | [[09 - The Physics Force Correctness Campaign]] | Ten defects found in my own assembled timestep |
| 10 | [[10 - Validation, Calibration and Results]] | Gates, the macrograph lock, held-out predictions, sweeps |
| 11 | [[11 - Software Engineering and Tooling]] | Taichi SoA, VRAM planning, viewer, VTK, HPC |
| 12 | [[12 - Limitations and Open Work]] | Where the twin is weak and what I would do next |

---

## The one-paragraph result

I calibrated against a single **ER70S-6 bead-on-plate macrograph** measuring **7.0 mm wide × 3.0 mm deep**. With arc efficiency $\eta = 0.72$ and Goldak semi-axes fitted, the twin predicts **6.8 × 3.2 mm**, an error of about **6.7 %**. Holding those knobs frozen, held-out process variants move in the physically correct direction: faster travel (11 mm/s) shrinks the pool to 2.40 × 0.80 mm, and higher current (120 A) grows it to 8.40 × 4.00 mm. Kernel-level benchmarks are tight — thermal diffusion within **1 %** of the analytical solution, Poiseuille within **8 %**, the Stefan front within **8 %**, VOF mass conserved to **0.5 %**.

> [!warning] The honest caveat, stated up front
> I have **one** experimental anchor. Fitting the model until one coupon matches is, as I put it in my own notes, like tuning a recipe until dinner tastes right once. The trend tests pass; the absolute-accuracy claim is still weak until I cut more coupons. See [[12 - Limitations and Open Work]].

---

## Scope boundaries I set deliberately

> [!info] Explicit non-goals
> I refused to model: grain structure and microstructure, residual-stress and distortion FEA, powder-bed DEM, laser ray-tracing, full arc-plasma MHD (the arc enters only as boundary fluxes), LES turbulence, and full G-code CAM. Every one of these was a real temptation and every one would have cost me the melt pool.

---

## Repository map

```
waam_twin/
├── twin.py              # WAAMTwin orchestrator, the public API
├── grid.py              # Structure-of-arrays Taichi fields, lattice units
├── lattice.py           # D3Q19 velocity set, VRAM estimator
├── kernels/             # All @ti.kernel GPU code lives here
│   ├── lbm.py           #   collision, streaming, clamps
│   ├── thermal.py       #   advection-diffusion, phase change, arc, evaporation
│   ├── vof.py           #   free surface, CSF, wetting, freeze/remelt
│   ├── weld.py          #   arc pressure, recoil, gas shear, droplet impact
│   ├── lorentz.py       #   electric potential solve, Ampère B_theta, JxB
│   └── grid_init.py     #   initialisation, deposition, moving window
├── solvers/coupled_step.py   # THE timestep — 12 stages, fixed order
├── physics/             # Host-side helpers and force closures
├── validation/          # ~90 regression modules + baselines
├── tools/              # Sweeps, ablation, calibration fitting, reports
├── viewer/             # Interactive Taichi GGUI melt-pool viewer
└── docs/               # Physics catalogue, formulae LaTeX, validation write-ups
```

---

## Related

- [[08 - What I Gathered From the Papers]] — the bibliography behind every equation
- `docs/WAAM_FORMULAE.tex` — my own equation catalogue with 46 bibliography entries
- `docs/WAAM_WELD_POOL_PHYSICS_CENTRE.md` — the master physics reference I maintained alongside the code
