# Reference case: ER70S-6 bead-on-plate (Park 2019 grounded lock)

New to macrographs / multipass / calibration vs held-out?
Start with [UNDERSTANDING_CALIBRATION_AND_DATA.md](UNDERSTANDING_CALIBRATION_AND_DATA.md).

Published catalog: [data/published_macrograph_catalog.json](data/published_macrograph_catalog.json).

## Locked twin job

| Item | Value |
|------|-------|
| Job | `jobs/examples/bead_calibrate.yaml` (HPC: `bead_calibrate_hpc.yaml`) |
| Material | `materials/validated/ER70S-6.v1.yaml` |
| Calibration overlay | `null` |
| Heat source | Goldak double-ellipsoid |
| Physics | `physics_tier: full` + soft-onset CC recoil |
| **Experimental anchor** | **Park et al., Appl. Sci. 2019** ([DOI 10.3390/app9214626](https://doi.org/10.3390/app9214626)) |

The previous uncited YAML lock (**7.0 × 3.0 mm**, `macrograph_ER70S-6_bead_on_plate`) is **retired**. Do not restore it.

## Experimental / literature reference (flat P-GMAW)

| Quantity | Value | Source |
|----------|-------|--------|
| Pool width | **6.1 mm** | Park 2019 etched macro (flat) |
| Pool depth | **1.8 mm** | same |
| Bead height | **2.2 mm** | same |
| Wire | ER70S-6 Ø1.0 mm | paper |
| Plate | SS400, 9 mm | paper |
| Travel | 10 mm/s (60 cm/min) | paper |
| WFR | 7 m/min | paper |
| CTWD | 15 mm | paper |
| Gas | 95% Ar + 5% CO₂ | paper |
| Current (twin) | ≈91 A mean | from published pulse 35/345 A @ ~151.5 Hz |
| Voltage (twin) | 22 V estimate | synergic; mean V not tabulated |

`claim.absolute_wd` stays **false** until `auto_calibrate` re-fits η/Goldak/recoil on the declared mesh tier against this reference.

## Held-outs (grounded or trend-only)

| Job | Role |
|-----|------|
| `bead_calibrate_heldout_fast.yaml` | Faster travel — **trend only** (no absolute macro) |
| `bead_calibrate_heldout_hot.yaml` | Higher I/WFS — **trend only** |
| `bead_calibrate_heldout_macro2.yaml` | Park **overhead** 5.7×1.9 mm (soft: position not modeled) |
| `bead_bruno_gmaw.yaml` | Figshare surface W/H (soft material) |
| `wall_pioneer_m1.yaml` | Zenodo PIONEER M1 remelt/wall |

```bash
PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.auto_calibrate \
  --job jobs/examples/bead_calibrate.yaml --write
PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.prediction_report --tier shop --with-bruno
```

## Fitted vs predicted

| Fitted (do not change on held-outs) | Predicted / model closures |
|-------------------------------------|----------------------------|
| η, Goldak a_f/a_r/b/c, σ, penetration | Lin–Eagar \(p_0(I,\sigma)\) |
| `evap_cooling_scale`, `C_acc`, `T_boil` | CC \(P_\mathrm{sat}(T)\), Lorentz J×B |
| I×V on the calibrate case | Marangoni from material \(d\gamma/dT\) + Sahoo |

## Related tests

- `test_calibrate_physics_credibility` / `test_calibrated_pool` — fitted W/D vs job `reference`
- `test_heldout_prediction` — lock + directional prediction
- Catalog + PIONEER/Bruno notes: [PIONEER_BRUNO_DATASET.md](PIONEER_BRUNO_DATASET.md)
