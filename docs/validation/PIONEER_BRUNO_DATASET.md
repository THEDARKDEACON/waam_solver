# External datasets (PIONEER / Bruno / CMT) — materials matter

Local root: `/mnt/shared_ntfs/PIONEER_PROJECT`  
Extracted numbers: [data/pioneer_bruno_metrics.json](data/pioneer_bruno_metrics.json)

## Material matrix (read this first)

| Dataset | Wire | Plate | Process | Match to `ER70S-6.v1` lock? | Twin use |
|---------|------|-------|---------|------------------------------|----------|
| **PIONEER M1** | **ER70S-6** (Böhler Q G 3) | steel plate 12 mm | GMAW-style WAAM wall | **Yes (class)** | `wall_pioneer_m1.yaml` |
| **PIONEER M2** | **ER110SG** (Union X85) | steel plate 12 mm | same | **No** | do not score with ER70S-6 lock |
| **Bruno GMAW** | “Carbon steel” Ø1 mm (grade not certified in `data.cnf`) | 6 mm carbon steel | conventional GMAW | **Soft / uncertain** | `bead_bruno_gmaw.yaml` surface gate only |
| **CMT profiles** | **ER100** Ø1.2 mm | **S235** 20 mm | **MIG CMT**, Ar82/CO2 18 | **No** | geometry/process only — **not** an ER70S-6 prediction |

**Rule:** held-outs that keep Goldak/η/recoil locked also keep the **material YAML** locked. If the coupon wire/plate is not ER70S-6 (or a documented mild-steel equivalent), treat results as exploratory — do not claim calibrate-lock prediction, and do not retune the lock to fit them.

## Wired jobs (ER70S-6 path)

| Job | Dataset | Status |
|-----|---------|--------|
| `jobs/examples/bead_bruno_gmaw.yaml` | Bruno surface W/H | **CLOSED** (soft material) |
| `jobs/examples/wall_pioneer_m1.yaml` | PIONEER M1_30 wall + remelt≈2.0 mm | **CLOSED** (ER70S-6) |

```bash
python3 -m waam_twin.tools.validation_gate_status
python3 -m waam_twin.tools.seed_goldak_from_pool --width 7 --depth 3
python3 -m waam_twin.tools.prediction_report --with-bruno
python3 -m waam_twin.tools.multipass_report --job jobs/examples/wall_pioneer_m1.yaml
```

PIONEER M1_30 remelt into plate digitized from `Macros/M1-30.JPG` (etch contrast, ~2.0 mm).

## CMT profile dataset (checked, not locked)

Path: `CMT_Profile_dataset/` (Recherche Data Gouv / ENSAM Metz — RobustAM).

- 140 beads, 4 parameter sets (Va × Vfil), Fronius TP-180i **CMT**
- CTWD 13 mm, wire **ER100**, plate **S235** 200×250×20 mm
- Rich laser profiles (`mesure_scan_nappe/RAM_WAAM_Mono_*.csv`) + generator logs

| Nj | Va (mm/s) | Vfil (m/min) | Typical I×V (arc-on) |
|----|-----------|--------------|----------------------|
| 1 | 13.3 | 9.0 | ~270 A × ~17 V |
| 2 | 18.3 | 7.5 | ~270 A × ~17 V |
| 3 | 8.3 | 10.0 | ~270 A × ~17 V |
| 4 | 18.3 | 10.0 | ~270 A × ~17 V |

**Why it is not wired into the ER70S-6 prediction suite:** different wire (ER100), CMT transfer, and much higher heat input than the 100 A / 6.5 mm/s lock. Useful later with an **ER100 / HSLA** material card, or for qualitative CMT bead-shape trends only.

## Still pending

1. ER70S-6 held-out macro at 5 mm/s → `bead_calibrate_heldout_macro2.yaml`  
2. Twolayer remelt/HAZ at **calibrate** process (100 A / 6.5 mm/s) — PIONEER remelt is a different window  
3. Bruno penetration D from macro / `met-data.dat`  
4. Crown FH-30 coupon + material promotion  
5. Optional ER100 material card before using CMT profiles  
6. Careful CMT profile → W/H pipeline (scans use `-99.9999` invalids)
