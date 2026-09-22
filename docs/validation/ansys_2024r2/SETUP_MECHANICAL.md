# Ansys 2024 R2 setup guide — bead-on-plate (Transient Thermal + Goldak)

This is the **hands-on setup** for a simple weld that mirrors
`jobs/examples/bead_on_plate.yaml`. Follow it in order in **Ansys Workbench 2024 R2**.

You will build:

- A **60 × 60 × 10 mm** steel plate  
- A **moving Goldak** heat source along Y = 20 → 40 mm at X = 30 mm  
- **2 seconds** of welding at **10 mm/s**, net power **≈ 981 W**  
- Results: **fusion width & depth** (macro) and a **temperature probe** (micro)

You will **not** model wire deposition or bead crown. That is intentional for a fair
thermal compare to `waam_twin` pool W/D.

Numbers: [`params_bead_on_plate.json`](params_bead_on_plate.json)  
APDL files: [`scripts/`](scripts/)

---

## Before you start

1. License: **Ansys Mechanical** (Transient Thermal). Fluent is not required.
2. Know where this folder lives on disk, e.g.  
   `...\WAAM\waam_twin\docs\validation\ansys_2024r2\`
3. In Workbench menu **Units**, set length preference to **mm** (Metric mm).  
   Temperature can stay **°C**.

---

## Step 1 — Create the project

1. Open **Ansys Workbench 2024 R2**.
2. In the Toolbox (left), expand **Analysis Systems**.
3. Drag **Transient Thermal** onto the white Project Schematic.
4. **File → Save As…**  
   Suggested name: `waam_bead_on_plate_tierA.wbpj`  
   Suggested folder: this package’s `workbook/` directory (results stay local).

You should see a system block with cells: Engineering Data, Geometry, Model, Setup, Solution, Results.

---

## Step 2 — Material (Engineering Data)

1. Double-click the **Engineering Data** cell.
2. In the outline, right-click **Data Sources** / empty material list → **Engineering Data Sources** if you need libraries; for a custom material:
3. Click **Click here to add a new material**, name it `ER70S-6_compare`.
4. From the Toolbox (left), drag onto the material:
   - **Density**
   - **Isotropic Thermal Conductivity**
   - **Specific Heat**
5. Enter constants (match the twin):

| Property | Value | Unit (pick in the property row) |
|----------|------:|----------------------------------|
| Density | **7000** | kg m^-3 |
| Isotropic Thermal Conductivity | **30** | W m^-1 C^-1 |
| Specific Heat | **680** | J kg^-1 C^-1 |

6. *(Optional but better for W/D)* Add latent heat between solidus and liquidus:
   - Solidus **1474.85 °C**, liquidus **1519.85 °C**, \(L = 2.72 \times 10^5\) J/kg  
   - Use **Enthalpy** table or **Specific Heat** with latent-heat option if your license UI shows it.
7. Close Engineering Data (return to Project Schematic). Material is stored in the project.

---

## Step 3 — Geometry (SpaceClaim)

1. Double-click the **Geometry** cell. SpaceClaim (or DesignModeler) opens.
2. Set units to **mm** if prompted (SpaceClaim: File → SpaceClaim Options → Units → Millimeters).
3. Create the plate:
   - **Design → Sketch** on the XY plane (or Front — whatever gives X–Y as the plate face).
   - Draw a **rectangle** from (0, 0) to (60, 60).
   - **Pull** (extrude) **10 mm** in +Z.  
     Result: solid from Z = 0 (bottom) to Z = 10 (top / weld face).
4. Optional named selections (helps later):
   - Select the **top** face (Z = 10) → right-click → **Named Selection** → `TopFace`
   - Bottom face → `BottomFace`
   - Four sides → `SideFaces`
5. **File → Save** / close SpaceClaim and return to Workbench. Geometry cell should show a green check / up-to-date.

**Torch path to remember (top face):**

| | X (mm) | Y (mm) | Z (mm) |
|--|-------:|-------:|-------:|
| Start (t = 0) | 30 | 20 | 10 |
| End (t = 2 s) | 30 | 40 | 10 |

Travel direction is **+Y**. Width is measured in **X**; depth in **−Z** from the top.

---

## Step 4 — Open Mechanical and assign material

1. Double-click the **Model** cell (opens **Mechanical**).
2. In the tree: **Geometry →** your solid body.
3. In Details (bottom left): **Assignment** → select `ER70S-6_compare`.
4. Confirm **Coordinate System** is the global Cartesian with origin at a plate corner (0,0,0) as modeled.  
   If your sketch was not at the origin, either remake geometry or adjust `X0`,`Y0`,`ZTOP` in the APDL params later.

---

## Step 5 — Mesh (important for a real pool)

1. Click **Mesh** in the tree.
2. Coarse default is **not** enough. Add sizing:

### 5.1 Body of Influence (weld band)

1. Right-click **Mesh → Insert → Method** or use **Sizing** + Body of Influence:
2. Easiest robust approach for 2024 R2:
   - Right-click **Mesh → Insert → Sizing**.
   - Scope: **Bodies** → select the plate (or later restrict with BOI).
   - Element Size: **0.5 mm** for a first run on the whole plate if the machine can take it.  
     If too heavy, use a **Body of Influence**:
     - Create a second solid in SpaceClaim: box roughly  
       X = 22–38 mm, Y = 12–48 mm, Z = 0–10 mm (covers the path ± margin).  
     - In Mechanical Mesh: **Insert → Body of Influence**, assign that body, Element Size **0.45 mm**.
     - On the plate body: Element Size **1.5–2.0 mm**.
3. Through-thickness: aim for about **20 elements** across 10 mm in the weld band (≥ 0.5 mm).
4. Click **Generate Mesh**. Note element count in the Mesh statistics (Details).

**Target:** weld-band edge length ≈ **0.4–0.5 mm** (same spirit as twin `dx_mm: 0.5`).

---

## Step 6 — Analysis settings (time)

1. In the tree expand **Transient Thermal**.
2. Click **Analysis Settings**.
3. Set:

| Field | Value |
|-------|------:|
| Number Of Steps | **2** (weld + cool-down) |
| Step 1 End Time | **2** s |
| Step 2 End Time | **7** s (5 s cool-down; adjust if you like) |
| Auto Time Stepping | **On** |
| Initial Time Step | **0.002** s |
| Minimum Time Step | **0.001** s |
| Maximum Time Step | **0.02** s |

4. Under **Output Controls**, set **Calculate Thermal Results** / store results every substep or every **0.05–0.1 s** so you can animate and pick t = 2 s.

---

## Step 7 — Initial temperature

1. Right-click **Transient Thermal → Insert → Initial Condition → Temperature**  
   (or **Initial Temperature** depending on UI wording).
2. Scope: **All Bodies**.
3. Value: **19.85 °C** (293 K ambient in the twin job).

---

## Step 8 — Convection (and radiation)

### Convection

1. Right-click **Transient Thermal → Insert → Convection**.
2. Scope: `TopFace` + `SideFaces` (and `BottomFace` if the coupon is open underneath).  
   If you have no named selections: select those faces in the graphics window.
3. Film Coefficient: **35** W/m²·°C  
4. Ambient Temperature: **19.85 °C**

### Radiation (recommended)

1. Insert → **Radiation**.
2. Same faces as convection (or top + sides).
3. Emissivity: **0.4** (engineering guess).
4. Ambient Temperature: **19.85 °C**.
5. Correlation: Ambient / To Ambient (default).

There is **no** structural support BC — this is thermal only.

---

## Step 9 — Goldak moving heat source (APDL Commands)

Workbench does not ship a “Goldak weld” button for this case. You inject heat with
**Commands (APDL)** using the macros in `scripts/`.

### 9.1 Copy the macro paths

On your machine, note the full paths to:

- `...\ansys_2024r2\scripts\goldak_load_params.mac`
- `...\ansys_2024r2\scripts\goldak_heat_source.mac`

Use **forward slashes** or doubled backslashes in APDL. Example Windows:

```text
D:\WAAM\waam_twin\docs\validation\ansys_2024r2\scripts
```

### 9.2 Commands that run every substep (weld)

1. Right-click **Transient Thermal → Insert → Commands (APDL)**.
2. Rename it (optional) to `Goldak_Moving`.
3. In Details of that Commands object:
   - **Step Selection / Step Number**: include **Step 1** (0–2 s).  
     Prefer **All** only if the heat-source macro zeros itself after `TWELD` (ours does).
4. In the Commands editor pane, paste:

```text
/INPUT,'D:/WAAM/waam_twin/docs/validation/ansys_2024r2/scripts/goldak_load_params','mac'
/INPUT,'D:/WAAM/waam_twin/docs/validation/ansys_2024r2/scripts/goldak_heat_source','mac'
```

Edit the path to **your** install location.  
Do **not** include the `.mac` extension in the `/INPUT` name field (APDL style: name, extension separate).

What the macros do:

- Net power `QNET = 0.72 × 131 × 10.4 ≈ 981.1 W`
- Torch at `X = 30`, `Y = 20 + 10·t` (mm, s)
- Double-ellipsoid Goldak sizes from the twin job
- After `t > 2 s`, heat generation is cleared

### 9.3 Optional: explicit cool-down clear

Insert a second **Commands (APDL)** scoped to **Step 2** only:

```text
BFE,ALL,HGEN,,0
```

### 9.4 If `/INPUT` is blocked

Paste the full contents of `goldak_load_params.mac` then `goldak_heat_source.mac`
into one Commands object (params first).

### 9.5 Unit check (do this once)

After a short test solve, if the plate barely heats or melts the whole coupon:

| Symptom | Likely cause |
|---------|----------------|
| Almost no heating | HGEN expected in **W/m³** but macro wrote **W/mm³** (or path wrong / Commands not in step) |
| Entire plate molten | Opposite units, or η applied twice, or mesh huge + wrong Q |

If HGEN must be W/m³ in your unit system, multiply `QV` by `1e9` in `goldak_heat_source.mac` (comment in that file).  
Verify with a **fixed hot spot** once before trusting W/D.

---

## Step 10 — Temperature probe (micro)

1. Right-click **Solution → Insert → Probe → Temperature**  
   (or **Result Probe** depending on 2024 R2 wording).
2. Location by coordinates: **X = 20 mm, Y = 10 mm, Z = 2 mm**  
   (from plate origin; Z from bottom — matches job probe intent).
3. After solve, right-click the probe → **Create Chart** / export tabular data for T(t).

From the chart you can read:

- Peak temperature  
- Rough \(t_{8/5}\) if the probe crosses 800 → 500 °C  
- Time spent above 800 °C / 1100 °C  

Those are your **micro** results.

---

## Step 11 — Solve

1. Click **Solve** (lightning bolt) on the toolbar, or right-click **Solution → Solve**.
2. Watch the progress monitor. First mesh at 0.5 mm weld band may take **tens of minutes**.
3. If it fails on Commands syntax, open the **Solution Information** sheet and search for APDL errors (`/INPUT` path, undefined parameters).

---

## Step 12 — Macro results (pool width & depth)

You want the **fusion zone** at (or near) end of weld: **t = 2.0 s**.

1. Under **Solution**, insert **Temperature**.
2. In Details, set **Display Time** to **2** s (or last substep of step 1).
3. Evaluate the result.
4. In the legend / contour controls, set a contour band or iso-surface at  
   **T = 1519.85 °C** (liquidus). Everything at or above that is “fused” for this compare.
5. Measure:
   - **Width W:** on a cross-section **perpendicular to travel** (cut at constant Y through the hottest pool), measure the X-extent of T ≥ T_liq.  
     Travel is +Y, so width is in **X**.
   - **Depth D:** on the centerline plane **X = 30 mm**, measure how far below Z = 10 the T ≥ T_liq region goes.

### Export for the Python helper (optional)

1. Export nodal coordinates + temperature at t = 2 s to CSV with columns:

```text
x_mm,y_mm,z_mm,T_C
```

(Mechanical: File → Export, or a small Mechanical Python snippet.)

2. On a machine with Python:

```bash
cd waam_twin/docs/validation/ansys_2024r2
python3 scripts/extract_fusion_zone.py \
  --csv /path/to/ansys_T.csv \
  --params params_bead_on_plate.json \
  --out comparison/ansys_metrics.json
```

Fill probe micro fields in `comparison/results_template.json` if you prefer manual entry.

---

## Step 13 — What “done” looks like

You should have:

| Result | How you got it |
|--------|----------------|
| Pool width W (mm) | Fusion isotherm at t = 2 s |
| Pool depth D (mm) | Same |
| Peak plate / probe T | Temperature plot + probe chart |
| Cooling / time-above-T | Probe chart |
| *(Not available)* Bead height | Tier A does not deposit metal |

Write W and D into `comparison/ansys_metrics.json` (from the template or extract script).

---

## Step 14 — Compare to waam_twin (after Ansys works)

Only after the Ansys model runs cleanly:

```bash
# From repo root — twin side
PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.ansys_compare_export \
  --validated-material \
  --out waam_twin/docs/validation/ansys_2024r2/comparison/twin_metrics.json

python3 waam_twin/docs/validation/ansys_2024r2/scripts/compare_to_twin.py \
  --ansys waam_twin/docs/validation/ansys_2024r2/comparison/ansys_metrics.json \
  --twin  waam_twin/docs/validation/ansys_2024r2/comparison/twin_metrics.json \
  --out   waam_twin/docs/validation/ansys_2024r2/comparison/report.md
```

Then use [`comparison/decision_rubric.md`](comparison/decision_rubric.md).

---

## Quick parameter cheat sheet

| Item | Value |
|------|------:|
| Plate | 60 × 60 × 10 mm |
| Path | (30,20,10) → (30,40,10) mm |
| Time | 0–2 s weld, optional cool-down |
| Speed | 10 mm/s |
| I, V, η | 131 A, 10.4 V, 0.72 → **981.1 W** |
| Goldak a_f, a_r, b, c | 2.2, 4.2, 3.0, 1.5 mm |
| Goldak f_f, f_r | 0.6, 1.4 |
| T_ambient | 19.85 °C |
| h_conv | 35 W/m²K |
| T_liq (fusion) | 1519.85 °C |
| Probe | (20, 10, 2) mm |

---

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| Commands ignored | Object under Transient Thermal? Correct step? Solve Information for APDL errors |
| `/INPUT` fails | Path wrong; use `/INPUT,'full/path/without_ext','mac'` |
| No melt pool | Mesh too coarse; HGEN units; QNET; T displayed in wrong unit |
| Pool huge | HGEN ×1e9 too large; ambient/radiation off by accident; latent heat missing so effective heat capacity low |
| Torch in wrong place | Geometry origin ≠ (0,0,0); edit `X0`,`Y0`,`ZTOP` in `goldak_load_params.mac` |
| Compare to twin bead height | Don’t — Tier A has no crown |

---

## Optional next

- Fluent melt-pool CFD: [`SETUP_FLUENT.md`](SETUP_FLUENT.md)  
- Package overview / fairness table: [`README.md`](README.md)
