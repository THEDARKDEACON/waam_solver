# Playbook — Ansys Tier A grounding (while HPC clamp runs)

**Purpose:** Use wait time on HPC clamp/GCI to ground the **thermal / Goldak**
problem for *your* bead-on-plate window in Ansys Mechanical 2024 R2, then compare
fairly to `waam_twin`. This does **not** clear Mach/force clamps and does **not**
replace Tier-F CFD.

**You will produce:** fusion W×D from \(T \ge T_\mathrm{liq}\), optional probe
cooling, a filled metrics JSON, a twin metrics JSON, and a short decision from the
rubric.

---

## Scope (read once)

| In scope (Tier A) | Out of scope |
|-------------------|--------------|
| Transient thermal + moving Goldak | Wire deposition / bead crown |
| Fusion isotherm W / D | Marangoni, Lorentz, recoil, VOF |
| Probe peak T / \(t_{8/5}\) | Mach / force clamp clearance |
| Ansys ↔ twin ↔ (optional) macrograph | Fluent melt-pool CFD (Tier B later) |

If Ansys and the twin differ on **bead height**, that is not an Ansys Tier A fail.

Locked numbers: [`params_bead_on_plate.json`](params_bead_on_plate.json)  
Click-by-click UI: [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)  
Decision rules: [`comparison/decision_rubric.md`](comparison/decision_rubric.md)

---

## Prerequisites

1. Ansys Workbench **2024 R2** with **Mechanical Transient Thermal** license.  
2. Clone of `waam_twin` with this folder readable from the Ansys machine (or copy
   `ansys_2024r2/` onto a USB / shared drive).  
3. Python 3.10+ with the twin installed (`pip install -e .`) for the twin export
   step (can be a different machine).  
4. Optional but valuable: one macrograph W×D on the **same** I, V, travel, stickout.

---

## Checklist (do in order)

### Phase 0 — Freeze inputs (5 min)

- [ ] Open [`params_bead_on_plate.json`](params_bead_on_plate.json) and confirm:

| Item | Value |
|------|------:|
| Plate | 60 × 60 × 10 mm |
| Path (top face) | (30, 20, 10) → (30, 40, 10) mm |
| Travel | 10 mm/s → **2.0 s** weld |
| \(I \times V \times \eta\) | 131 A × 10.4 V × 0.72 → **≈ 981.1 W** |
| Goldak \(a_f,a_r,b,c\) | 2.2, 4.2, 3.0, 1.5 mm |
| Goldak \(f_f,f_r\) | 0.6, 1.4 |
| \(T_\mathrm{liq}\) (fusion) | **1519.85 °C** (1793 K) |
| \(h_\mathrm{conv}\) | 35 W/m²·K |
| Ambient | 19.85 °C |

- [ ] Do **not** retune Goldak or η to chase a pretty Ansys W/D before the first
  compare — shared knobs must match the twin job
  (`jobs/examples/bead_on_plate.yaml`).

### Phase 1 — Build & solve Ansys (Workbench)

Follow [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md) steps 1–12. Condensed:

1. **Transient Thermal** system → save under `workbook/`
   (`waam_bead_on_plate_tierA.wbpj`).  
2. Material `ER70S-6_compare`: ρ=7000, k=30, cp=680; prefer latent heat
   solidus/liquidus as in params JSON.  
3. Geometry: 60×60×10 mm plate, origin at corner, top = weld face Z=10.  
4. Mesh: ≤ **0.5 mm** in a weld band around the path (whole-plate 0.5 mm OK if
   RAM allows).  
5. Initial T = ambient; convection h=35 on exposed faces; radiation optional
   (ε≈0.4).  
6. Commands (APDL) under Transient Thermal: load
   [`scripts/goldak_load_params.mac`](scripts/goldak_load_params.mac) then
   [`scripts/goldak_heat_source.mac`](scripts/goldak_heat_source.mac)
   (paths in SETUP guide).  
7. Analysis settings: end time **≥ 2 s** (add cool-down if you want \(t_{8/5}\)).  
8. Solve. Confirm HGEN / torch motion in Solve Information (no APDL path errors).

**Sanity before measuring**

- [ ] Peak T on the path is well above \(T_\mathrm{liq}\) (not “barely warm”).  
- [ ] Fusion blob moves with +Y torch, not stuck at t=0.  
- [ ] If pool is absurdly huge/tiny: check QNET, mesh, units (°C vs K), latent heat
  (see SETUP troubleshooting table).

### Phase 2 — Extract Ansys fusion W×D

At **t = 2.0 s** (end of weld):

1. Temperature result → isotherm / contour at **1519.85 °C**.  
2. Measure **width** = X extent of fused region at mid-path (or max envelope —
   note which).  
3. Measure **depth** = distance from top face down in −Z.  
4. Optional: result probe at (20, 10, 2) mm → peak T, \(t_{8/5}\), time above 800/1100 °C.

**Automated path (preferred if you export nodal CSV):**

```bash
cd waam_twin/docs/validation/ansys_2024r2
python3 scripts/extract_fusion_zone.py \
  --csv /path/to/ansys_T.csv \
  --params params_bead_on_plate.json \
  --out comparison/ansys_metrics.json
```

**Manual path:** copy [`comparison/results_template.json`](comparison/results_template.json)
→ `comparison/ansys_metrics.json` and fill:

```json
"macro": {
  "pool_width_mm": <W>,
  "pool_depth_mm": <D>,
  "peak_temp_C": <...>,
  "definition": "T >= T_liquidus isotherm"
}
```

Also set `checks.Q_net_W_verified`, `latent_heat_enabled`, `radiation_enabled`.

If you have a lab macrograph for this window, put measured W/D under
`experiment` in the same JSON (default template already has Park 6.1×1.8 as an
optional placeholder — replace with **your** coupon if different).

### Phase 3 — Twin metrics (same process)

On a machine with the twin + GPU/CPU:

```bash
cd waam_twin
export PYTHONPATH=..
export WAAM_BACKEND=cuda   # or cpu for a slow smoke

python3 -m waam_twin.tools.ansys_compare_export \
  --validated-material \
  --out docs/validation/ansys_2024r2/comparison/twin_metrics.json
```

(`ansys_compare_export` runs the locked bead job and writes Ansys-comparable
macro fields. Use `--help` for job overrides.)

Confirm twin JSON has non-null `macro.pool_width_mm` / `pool_depth_mm` and a hot
peak T (not ambient).

### Phase 4 — Compare & decide

```bash
cd waam_twin/docs/validation/ansys_2024r2
python3 scripts/compare_to_twin.py \
  --ansys comparison/ansys_metrics.json \
  --twin  comparison/twin_metrics.json \
  --out   comparison/report.md \
  --tol_pct 15
```

Open [`comparison/decision_rubric.md`](comparison/decision_rubric.md) and pick
**one** row:

1. Ansys ≈ experiment; twin also / worse / better → optional / pause / continue  
2. Ansys misses experiment; twin hits → **strong case for twin** (or Fluent B)  
3. Both miss experiment → fix shared η/Goldak/material/BC first  
4. Ansys ≈ twin, both miss experiment the same way → shared thermal setup bottleneck  
5. You care about crown / toe / multipass → Tier A cannot answer; keep twin

Fill the report sentence from the rubric into `comparison/report.md` or your logbook.

### Phase 5 — What to do next (parallel tracks)

| Outcome | Next action |
|---------|-------------|
| Tier A done; HPC clamp still running | Stop here; paste W/D + rubric row into verification notes |
| Clamps clear on HPC later | Re-run twin export at clear dx; keep Ansys Tier A as thermal anchor |
| Ansys ≪ experiment on W/D | Do not blame TRT/Mach — retune shared Goldak/η with experiment first |
| Need flow physics | Only then open [`SETUP_FLUENT.md`](SETUP_FLUENT.md) (Tier B) |

---

## Fairness rules (do not violate)

1. Same \(Q_\mathrm{net}\), Goldak axes, plate, material constants, \(h_\mathrm{conv}\).  
2. Fusion definition = **liquidus isotherm** on both sides for this compare.  
3. Never score bead height / toe against Tier A.  
4. Never raise twin Mach/force clamps to “match Ansys.”  
5. Shop twin may still be `FAIL_CLAMPS` on Mach — still valid for **thermal**
   envelope compare if you document operating regime.

---

## Artifacts to keep

| File | Role |
|------|------|
| `workbook/*.wbpj` (+ results) | Ansys project (usually gitignored) |
| `comparison/ansys_metrics.json` | Ansys macro/micro |
| `comparison/twin_metrics.json` | Twin export |
| `comparison/report.md` | Auto table + your rubric sentence |

Suggested commit (metrics + report only, not the Workbench binary):

```bash
git add docs/validation/ansys_2024r2/comparison/ansys_metrics.json \
        docs/validation/ansys_2024r2/comparison/twin_metrics.json \
        docs/validation/ansys_2024r2/comparison/report.md
```

---

## Time budget (typical)

| Phase | Wall time |
|-------|-----------|
| Phase 0 | 5 min |
| Phase 1 first build | 1–3 h (mesh/APDL debugging) |
| Phase 1 re-solve | 10–40 min |
| Phase 2–4 | 30–60 min |

---

## Related docs

- Package overview: [`README.md`](README.md)  
- Mechanical detail: [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md)  
- Optional Fluent: [`SETUP_FLUENT.md`](SETUP_FLUENT.md)  
- HPC clamp clearance (parallel track): [`../../HPC_CLAMP_CLEARANCE.md`](../../HPC_CLAMP_CLEARANCE.md)
