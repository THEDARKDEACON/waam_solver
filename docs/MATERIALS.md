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

Calibrated alloys live under `materials/validated/` with `status: calibrated`.

**ER70S-6.v1** is the locked material for `jobs/examples/bead_calibrate.yaml`
(Goldak GMAW window, macrograph ~7×3 mm). Prefer that job’s embedded η —
do **not** stack the legacy `materials/calibration/ER70S-6.bead_on_plate.yaml`
(η=0.65) on the locked calibrate case.

**ER70S-6.v2** adds denser literature `cp`/`k`/`μ` tables (`notes` field documents
sources). Do not point the calibrate job at v2 until `prediction_report` and
FULL validation are re-run.

```yaml
# jobs/examples/bead_calibrate.yaml
material: materials/validated/ER70S-6.v1.yaml
calibration: null
process:
  arc_efficiency: 0.72
```

For older Gaussian smoke jobs that still use a process overlay:

```yaml
material: materials/validated/ER70S-6.v1.yaml
calibration: materials/calibration/ER70S-6.bead_on_plate.yaml
```

## Wire vs plate (dissimilar coupon)

Top-level `material:` is the **wire / deposit**. Optional `plate.material:` is the **substrate**. Omit it (or set the same alloy) and the solver stays on the single-alloy path — `bead_calibrate.yaml` is unchanged.

```yaml
material: materials/placeholders/ER70S-6.yaml          # wire
plate:
  thickness_mm: 10.0
  size_mm: [50, 50]
  material: materials/validated/SS316L.v1.yaml         # substrate
```

Plate cells are tagged at init (`alloy_id=1`, `alloy_frac=1`) and keep that birth
tag when they melt. Gas→metal deposit is wire (`alloy_id=0`, `alloy_frac=0`).
Set `simulation.enable_alloy_mixing: true` to Jacobi-average `alloy_frac` among
neighbouring liquid cells (dilution). Birth `alloy_id` is unchanged; properties
lerp on `alloy_frac`. Mixing is **off** by default so the ER70S-6 calibrate lock
is unchanged.

Thermal `k`, `cp`, melting range, and latent heat are per cell. LBM still uses lattice density 1; hydrodynamic `force_scale` for the pool is the wire density. VTK `Alloy_id` is 0=wire, 1=plate.

See `jobs/examples/bead_dissimilar_ss316l_plate.yaml`.

## Promotion workflow (placeholder → calibrated)

1. **Copy** `materials/placeholders/<alloy>.yaml` → `materials/validated/<alloy>.v1.yaml`.
2. **Set** `status: calibrated` and document data sources / process window in `notes`.
3. **Add** optional `tables` block (cp, k, μ, dγ/dT) from literature or measurement.
4. **Lock** η / Goldak / recoil in the job YAML (preferred) or create a calibration overlay.
5. **Add** `reference` (experiment) and `model_reference` (simulator envelope) to the job.
6. **Run** `WAAM_FULL_VALIDATION=1` + held-out prediction report; confirm pass.
7. **Bump** material version (`v1` → `v2`) when properties change; never overwrite validated files in place.

Until step 6 passes, keep `status: placeholder` and expect startup warnings.

See also: [validation/reference_case_ER70S6.md](validation/reference_case_ER70S6.md).
