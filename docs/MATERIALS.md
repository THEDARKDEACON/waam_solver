# Materials (waam_twin v2)

Materials are **data**, not hardcoded physics. Each alloy is a YAML file with thermophysical properties used by the enthalpy–porosity and LBM solvers.

## Status field

| Status | Meaning |
|--------|---------|
| `placeholder` | Literature-order-of-magnitude values for development only. A warning is printed at load time. |
| `calibrated` | Tuned against experiments or reference simulations for a specific process window. |

Built-in presets (`ER70S-6`, `SS316L`, etc.) resolve to `materials/placeholders/` until validated files exist under `materials/validated/`.

## Required properties

| Key | Unit | Used for |
|-----|------|----------|
| `rho` | kg/m³ | LBM density, buoyancy |
| `cp` | J/(kg·K) | Enthalpy ↔ temperature |
| `L_fusion` | J/kg | Mushy-zone latent heat |
| `T_solidus`, `T_liquidus` | K | Phase change |
| `alpha` | m²/s | Thermal diffusion |
| `beta_T` | 1/K | Thermal expansion (buoyancy) |
| `dgamma_dT` | N/(m·K) | Marangoni driving force (overridden by surfactant model) |
| `surface_tension` / `gamma_0` | N/m | CSF reference; Sahoo model resets from \(\gamma(T_{\ell},a_S)\) |

## Surfactant block

```yaml
surfactant:
  model: sahoo        # or heiple (default)
  sulphur_ppm: 28.0
```

| Model | Behaviour |
|-------|-----------|
| `heiple` | Static S-ppm scale of YAML `dgamma_dT` (Mills/Heiple thresholds) |
| `sahoo` | Sahoo–DebRoy–McNallan (1988) Fe–S: builds a local \(\mathrm{d}\gamma/\mathrm{d}T(T,a_S)\) table for GPU property refresh |

Validated ER70S-6 uses `sahoo`. See `physics/surfactant.py` and `PHYSICS_FORCE_CORRECTNESS_SPEC.md` §5.9.

## Loading

```python
from waam_twin.materials import load_material

mat = load_material("materials/placeholders/ER70S-6.yaml")
print(mat.status)  # placeholder
```

Environment override: `WAAM_MATERIAL=path/to/alloy.yaml`

## Calibration overlays

Process-specific tuning (arc efficiency, heat-loss factor, Marangoni scale) lives in separate calibration YAML files loaded via `waam_twin.calibration.load_calibration()`. These **do not** replace material properties; they scale boundary/process terms for a given machine and shielding gas.

## Temperature-dependent tables (Phase 1)

Optional `tables` block with piecewise-linear knots `[T_K, value]`:

```yaml
tables:
  cp:
    - [300, 650]
    - [1793, 720]
  k:
    - [300, 28]
    - [1793, 34]
```

Access in Python via `mat.cp_at(T)`, `mat.k_at(T)`, etc.

When a YAML file includes `tables.cp` or `tables.k`, `WAAMTwin` uploads knots to GPU and uses per-cell `cp(T)`, `k(T)` during diffusion, phase recovery, and boundary losses.

## Validated materials

Calibrated alloys live under `materials/validated/` with `status: calibrated`. Pair with a process overlay in `materials/calibration/`:

```yaml
# jobs/examples/bead_on_plate.yaml
material: materials/validated/ER70S-6.v1.yaml
calibration: materials/calibration/ER70S-6.bead_on_plate.yaml
heat_loss:
  convection: true
  h_conv: 35.0
```

## Promotion workflow (placeholder → calibrated)

1. **Copy** `materials/placeholders/<alloy>.yaml` → `materials/validated/<alloy>.v1.yaml`.
2. **Set** `status: calibrated` and document data sources in a `notes` field.
3. **Add** optional `tables` block (cp, k, μ, dγ/dT) from literature or measurement.
4. **Create** `materials/calibration/<alloy>.<process>.yaml` with fitted η, `arc_sigma_scale`, etc. (`tools/fit_calibration`).
5. **Add** `model_reference` W/D to the matching job YAML (simulator envelope, not macrograph).
6. **Run** `WAAM_FULL_VALIDATION=1` + `WAAM_STANDARD_VALIDATION=1`; confirm pass via `python -m waam_twin.validation.run_all`.
7. **Bump** material version (`v1` → `v2`) when properties or calibration change; never overwrite validated files in place.

Until step 6 passes, keep `status: placeholder` and expect startup warnings.
