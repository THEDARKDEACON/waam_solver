# Verification

This document is the living verification plan for `waam_twin` after the
melt-pool rebuild (Phases 0–10).

## Ground rules

1. Never retune a physical knob to make a gate pass.
2. Any run with force or Mach clamps firing is **invalid** for validation.
3. Compare only at equal physical time or detected quasi-steady state.
4. All inputs are SI; lattice conversion is only in `waam_twin/units.py`.

## Operator ladder

| Gate | Module | Pass criterion |
|------|--------|----------------|
| Thermal diffusion | `test_thermal_diffusion` | L2 vs analytic within tol |
| Poiseuille | `test_lbm_poiseuille` | parabolic profile; `periodic_x=1` |
| Cavity | `test_lbm_cavity` | steady Re match |
| Stefan | `test_stefan_solidification` | front position |
| VOF mass | `test_vof_mass` | φ conservation |
| Laplace CSF | `test_laplace` | ΔP = γκ |
| Production τ | `test_production_tau` | finite fields at τ≈0.5007 |

## Critical-bug regressions (Phase 1)

| ID | Test | Pass criterion |
|----|------|----------------|
| C1 | `test_substrate_flow` | melted substrate \|u\| > 0 |
| C4 | `test_wetting_no_volume` | wetting BC Δm = 0 |
| H6 | `test_enthalpy_cap_init` | step-1 cap ≪ plate enthalpy |

## Mesh / GCI (Phase 6)

```
PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.mesh_convergence_report \
  --job jobs/examples/bead_calibrate.yaml --dx 0.40,0.30,0.25
```

- Equal travel distance (C3), never equal step counts.
- Clamps → `FAIL_CLAMPS`.
- Roache GCI ≤ 5 % on W and D → `PASS_GCI`.

## Diagnostics

```
python -m waam_twin.tools.diagnostics.pool_state --steps 6000
python -m waam_twin.tools.diagnostics.energy_ledger --steps 2000
python -m waam_twin.tools.diagnostics.profile_step --warm 500 --n 200 --prof 1
```

## Pre-fix results

All calibration / mesh / sensitivity results produced **before** the C1/C4/H6
fixes are marked **invalid**. See `docs/report/12 - Limitations and Open Work.md`.
