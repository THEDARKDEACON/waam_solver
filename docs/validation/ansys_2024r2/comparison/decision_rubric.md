# Decision rubric — Ansys vs experiment vs waam_twin

Use after Tier A Ansys + a twin run on the **same** process/Goldak/material constants.
Optional: paste measured macrograph W/D into `experiment` in the metrics JSON.

## Metrics that count for the decision

| Metric | Role |
|--------|------|
| Fusion / pool **W** and **D** | Primary macro gate |
| Probe peak T / \(t_{8/5}\) | Micro / HAZ sanity |
| Bead height / toe | Twin-only; **not** an Ansys Tier A fail |

Default agree band vs twin: **±15%** on W and on D  
(adjust if your lab uncertainty is larger).

## Rubric

### 1. Ansys ≈ experiment (macrograph) within band

| Twin vs experiment | Interpretation | Pursue waam_twin? |
|--------------------|----------------|-------------------|
| Also within band | Both codes capture thermal envelope | **Optional** — twin only if you need free surface, multiphysics, or speed/automation |
| Twin worse | Twin extras not helping W/D yet | **Pause twin as W/D predictor**; keep for R&D or fix freeze/wetting/dx first |
| Twin better | Flow/free-surface adding value | **Yes** for geometry-critical work |

### 2. Ansys disagrees with experiment; twin agrees

→ **Strong case for continuing waam_twin** (or Fluent Tier B).  
Conduction Goldak alone is insufficient for your penetration/width physics.

### 3. Both disagree with experiment

→ Fix shared inputs first (η, Goldak sizes, material, BC), not the twin architecture.  
Do not use this case to abandon or bless the twin.

### 4. Ansys ≈ twin, both miss experiment the same way

→ Shared Goldak/thermal setup is the bottleneck. Twin multiphysics is **not yet**
proven necessary until a held-out or a physics toggle (Marangoni, etc.) moves you
toward the macrograph.

### 5. Ansys ≈ twin on W/D, but you care about bead crown / toe / multipass remelt

→ Tier A cannot answer. **Continue twin** (or invest in Fluent VOF+deposition).

## Minimum experiment to ground the compare

1. One macrograph cross-section → measured W, D (and bead height if available).  
2. Ideally one thermocouple under/ beside the path.  
3. Same I, V, travel, stickout as `params_bead_on_plate.json`.

Without (1), you can still compare **Ansys ↔ twin**, but you cannot decide
“which code is right” — only “do they implement the same thermal problem.”

## Suggested report sentence

> On the bead-on-plate process window (131 A × 10.4 V, η=0.72, 10 mm/s),  
> Ansys 2024 R2 Transient Thermal Goldak gave W×D = … mm; waam_twin gave … mm;  
> macrograph … mm. Relative Ansys–twin errors were …%.  
> Decision: [continue / pause / optional] waam_twin because [rubric row].
