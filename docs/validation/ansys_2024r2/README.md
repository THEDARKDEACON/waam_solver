# Ansys 2024 R2 vs waam_twin — bead-on-plate comparison

**Start here (operator playbook):** [`PLAYBOOK_TIER_A.md`](PLAYBOOK_TIER_A.md)  
(checklist: Ansys build → extract W/D → twin export → compare → rubric)

**Click-by-click Workbench:** [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)

**Goal:** Run the **Park 2019 flat P-GMAW** bead-on-plate window in Ansys that mirrors
`jobs/examples/bead_calibrate.yaml`, extract **macro** (pool W×D) and **micro**
(HAZ / cooling) metrics, compare to the published macrograph **6.1 × 1.8 mm**,
and decide whether continuing `waam_twin` is necessary.

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
  PLAYBOOK_TIER_A.md                 ← end-to-end checklist (start here)
  params_bead_on_plate.json          ← numbers locked to bead_calibrate.yaml / Park 2019
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

1. Follow **[`PLAYBOOK_TIER_A.md`](PLAYBOOK_TIER_A.md)** end-to-end (or the detailed UI guide [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)).
2. Fill `comparison/ansys_metrics.json` (template or `scripts/extract_fusion_zone.py`).
3. Export twin metrics (`python -m waam_twin.tools.ansys_compare_export`) and run `scripts/compare_to_twin.py`.
4. Decide with [`comparison/decision_rubric.md`](comparison/decision_rubric.md).

---

## Process numbers (summary)

| Item | Value |
|------|-------|
| Plate | 60 × 60 × **9** mm (flat — no crown) |
| Path | (30, 20) → (30, 40) mm @ 10 mm/s → **2.0 s** weld |
| \(I \times V\) | **91 A × 22 V** |
| \(\eta\) | **0.82** → **\(Q_\mathrm{net} \approx 1642\) W** |
| Goldak | \(a_f=2.0\), \(a_r=3.8\), \(b=2.8\), \(c=2.6\) mm; \(f_f=0.6\), \(f_r=1.4\) |
| Macrograph | Park flat **6.1 × 1.8** mm (DOI 10.3390/app9214626); H=2.2 twin/exp only |
| Twin job | `jobs/examples/bead_calibrate.yaml` |
| \(T_\mathrm{liq}\) | 1793 K (1520 °C) for fusion isotherm |
| \(h_\mathrm{conv}\) | 35 W/m²K |

---

## Licensing note

Tier A needs **Ansys Mechanical** (Transient Thermal).  
Tier B needs **Fluent** (+ melting/solidification; VOF if you want free surface).  
Additive Suite DED is a different workflow and is **not** required for this compare.
