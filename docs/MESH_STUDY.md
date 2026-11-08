# Mesh study: how much does dx change the bead?

Goal: one physical calibration that holds across meshes, with a stated mesh
error, instead of a lock that is only valid at dx = 0.4 mm.

## Two tools

| Tool | What it checks | Where it runs |
|---|---|---|
| `validation/test_mesh_sensitivity.py` (core suite) | Thermal Goldak fusion W/D at dx 0.4 vs 0.2 mm within 5 % | CPU, ~15 s |
| `tools/mesh_study.py` | Full coupled bead (FSLBM, Marangoni, Lorentz…) at several dx, with GCI | GPU, hours |

The suite test catches mesh-dependent *modelling* errors (cell-unit
parameters, off-by-one-cell sources). Before the Nov 2026 Goldak fix it
showed 9–11 % W/D change between 0.4 and 0.2 mm; after it 0.8 % / 2.6 %.

## Running the study

```bash
# from the parent of waam_twin/
export PYTHONPATH=$PWD WAAM_BACKEND=cuda
python -m waam_twin.tools.mesh_study --dry-run             # grids, GPU memory
python -m waam_twin.tools.mesh_study --dx 0.5,0.4,0.3 --out runs/mesh_study/park.jsonl
python -m waam_twin.tools.mesh_study --dx 0.25,0.2   --out runs/mesh_study/park.jsonl   # overnight
python -m waam_twin.tools.mesh_study --report runs/mesh_study/park.jsonl
```

Meshes already in `--out` are skipped, so the study can be built up over
several sessions. Steady_bead options go after `--` (for example
`-- --cs 0.12` for a Smagorinsky sensitivity run; use a different `--out`).

### GPU memory (RTX 3050 4 GB, measured, `bead_calibrate.yaml` domain)

| dx (mm) | cells | peak GPU | approx. wall time for 2.5 s |
|---|---|---|---|
| 0.5 | 0.64 M | ~0.35 GB | ~12 min |
| 0.4 | 1.24 M | 0.53 GB | ~25 min |
| 0.3 | 2.94 M | 1.07 GB | ~1.3 h |
| 0.25 | 5.12 M | 1.78 GB | ~3 h |
| 0.2 | 10.0 M | 3.34 GB | ~7 h |

0.2 mm fits with ~750 MiB to spare: close other GPU applications.

## Reading the report

For W, D, H_frozen and deposited area the report gives the observed order
of convergence p, the extrapolated (dx → 0) value and the fine-grid GCI
(Celik et al. 2008), using the three finest meshes, and the error of the
target mesh (default 0.4 mm) against the extrapolated value. The gate
passes when W and D at the target are within `--gate-pct` (default 10 %).

Per-mesh flags say when a mesh's result is set by numerics rather than
resolution:

- **CLAMP**: Mach or force clamp on > 1 % of steps. The flow speed is capped
  at Ma·dx/dt (0.8 m/s at Ma 0.08), so W/D reflect the cap.
- **LES**: Smagorinsky eddy viscosity ≥ molecular (ν_t/ν > 1). The resolved
  flow then depends on Cs·dx and will drift with dx by construction.
- **TAU**: effective relaxation time 0.5 + 3ν(1 + ν_t/ν)dt/dx² < 0.505
  (marginal for TRT). τ grows as dx shrinks at fixed Ma, so coarse meshes
  are the most marginal.
- **UNSTEADY**: the slice window had not frozen.

A short GPU smoke run (1 s, 0.5 and 0.4 mm, Nov 2026) already showed CLAMP
(~9 % of steps) and ν_t/ν ≈ 1–1.2 on both meshes, and a deposited-area
ratio of 0.50 at 0.5 mm vs 0.86 at 0.4 mm. Expect the coarse meshes to be
flagged; the useful sequence is likely 0.3 / 0.25 / 0.2.

## Darcy across meshes

`C_darcy` is a lattice value at `C_darcy_ref_dt_s` and is rescaled by
dt/dt_ref, which keeps the SI damping constant across meshes, so the study
compares like with like. The lock value (160000 at 40 µs) is ≈ 3e13
kg m⁻³ s⁻¹ in SI: it stops all flow in the mush (f_l < 1). That is a
modelling choice to revisit during recalibration, not a mesh effect.

## Suggested acceptance

- Thermal suite test: ≤ 5 % (in place).
- Coupled study: W and D at the calibration mesh within 10 % of the
  extrapolated value, on meshes not flagged CLAMP. If 0.4 mm cannot meet
  that, calibrate on 0.3 mm and quote the 0.4 mm error as its uncertainty.
