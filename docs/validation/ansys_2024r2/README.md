# Ansys 2024 R2 vs waam_twin — bead-on-plate comparison

**Start here for setup:** [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)  
(click-by-click Transient Thermal + Goldak in Workbench 2024 R2)

**Goal:** Run a simple GMAW-style bead-on-plate case in Ansys that mirrors
`jobs/examples/bead_on_plate.yaml`, extract **macro** (pool W×D) and **micro**
(HAZ / cooling) metrics, and decide whether continuing `waam_twin` is necessary.

**Ansys version:** 2024 R2  
**Params:** [`params_bead_on_plate.json`](params_bead_on_plate.json)

---

## What you can (and cannot) compare fairly

| Quantity | Ansys Tier A (Mechanical Transient Thermal + Goldak) | waam_twin (full physics) | Fair? |
|----------|------------------------------------------------------|--------------------------|-------|
| Fusion zone width / depth from \(T \ge T_\mathrm{liq}\) | Yes (isotherm envelope) | Yes (liquid / remelted metal) | **Yes — primary gate** |
| Peak temperature / TC-style probe traces | Yes | Yes | **Yes** |
| Cooling rate / \(t_{8/5}\) / time-above-\(T\) | Yes (Eulerian nodal) | Yes (Eulerian; twin also has tracers) | **Mostly** |
| Bead crown height / free-surface toe | No (no deposited bead geometry) | Yes (VOF + deposition) | **No** |
| Marangoni / Lorentz / recoil pool shape | No | Yes | **No** unless Fluent Tier B |

**Recommendation:** Use **Tier A** for the go/no-go decision on “do we need a custom twin?”  
If Ansys isotherm W/D already matches your macrograph within your tolerance **and**
held-out process changes track experiment, a full free-surface twin is optional R&D.  
If Ansys matches isotherms but **misses bead height / toe / penetration under
parameter changes**, that is where `waam_twin` (or Fluent Tier B) earns its keep.

Wire feed / droplet momentum / free surface are **not** in Tier A — do not score
the twin as “wrong” for differing bead crown when Ansys never modeled it.

---

## Package layout

```
ansys_2024r2/
  README.md                          ← this file
  params_bead_on_plate.json          ← numbers locked to bead_on_plate.yaml
  SETUP_MECHANICAL.md                ← Tier A Workbench steps (2024 R2)
  SETUP_FLUENT.md                    ← Tier B optional CFD outline
  scripts/
    goldak_heat_source.mac           ← APDL Goldak volumetric source
    goldak_load_params.mac           ← loads params into APDL scalars
    extract_fusion_zone.py           ← W/D from exported nodal T CSV
    compare_to_twin.py               ← merge Ansys + twin metric JSONs
  comparison/
    metrics_schema.json              ← required fields for both codes
    results_template.json            ← fill after each run
    decision_rubric.md               ← how to interpret the comparison
```

---

## Quick start (Tier A)

1. Follow the setup guide: **[`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)** (Workbench → mesh → Goldak Commands → solve → measure W/D).
2. Fill `comparison/ansys_metrics.json` (template or `scripts/extract_fusion_zone.py`).
3. Export twin metrics and compare (commands in the setup guide, Step 14).
4. Read [`comparison/decision_rubric.md`](comparison/decision_rubric.md).

---

## Process numbers (summary)

| Item | Value |
|------|-------|
| Plate | 60 × 60 × 10 mm |
| Path | (30, 20) → (30, 40) mm, 20 mm @ 10 mm/s → **2.0 s** weld |
| \(I \times V\) | 131 A × 10.4 V |
| \(\eta\) | 0.72 → **\(Q_\mathrm{net} \approx 981\) W** |
| Goldak | \(a_f=2.2\), \(a_r=4.2\), \(b=3.0\), \(c=1.5\) mm; \(f_f=0.6\), \(f_r=1.4\) |
| \(T_\mathrm{liq}\) | 1793 K (1520 °C) for fusion isotherm |
| \(h_\mathrm{conv}\) | 35 W/m²K |

---

## Licensing note

Tier A needs **Ansys Mechanical** (Transient Thermal).  
Tier B needs **Fluent** (+ melting/solidification; VOF if you want free surface).  
Additive Suite DED is a different workflow and is **not** required for this compare.
