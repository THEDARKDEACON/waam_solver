# Tier B — Ansys Fluent 2024 R2 (optional melt-pool CFD)

Use only if Tier A isotherms already look reasonable and you need **closer**
physics to `waam_twin` (buoyancy / optional Marangoni / melting).  
This is a **multi-day** setup, not the primary go/no-go path.

## Scope vs twin

| Feature | Fluent Tier B (typical) | waam_twin |
|---------|-------------------------|-----------|
| Melting / solidification (enthalpy-porosity) | Yes | Yes |
| Buoyancy (Boussinesq) | Yes | Yes |
| Marangoni (surface tension gradient BC) | Possible | Yes |
| Free-surface VOF + wire deposition | Hard / research-grade | Built-in |
| Lorentz / recoil / Lin–Eagar pressure | Custom UDF | Built-in |
| Goldak volumetric heating | UDF or source term | Built-in |

**Fair Fluent vs twin compare:** same plate, path, \(Q_\mathrm{net}\), \(T_\mathrm{liq}\),  
report W/D from liquid fraction \(f_l \ge 0.5\) or \(T \ge T_\mathrm{liq}\).  
Do **not** require Fluent to reproduce twin bead crown unless you invest in VOF+mass source.

## Suggested Fluent workflow (outline)

1. **Workbench** → Fluent (with Fluent Meshing) or standalone Fluent 2024 R2.
2. Geometry: same 60×60×10 mm plate; fluid zone = metal domain (solid region treated via mushy zone).
3. Mesh: 0.3–0.5 mm in pool; inflation not required for first pass.
4. Models:
   - Energy on  
   - Solidification/Melting on (\(T_\mathrm{sol}\), \(T_\mathrm{liq}\), \(L\), mushy zone parameter)  
   - Optional: Viscous laminar + Boussinesq  
   - Optional: Multiphase VOF (only if modeling free surface)
5. Materials: steel props from `params_bead_on_plate.json`.
6. Cell zone source: Goldak heat generation via **UDF** (adapt Mechanical macros’ formula to C UDF `DEFINE_SOURCE`).
7. BCs: wall convection \(h=35\); top surface Marangoni shear if enabled.
8. Transient: 2 s weld @ 10 mm/s; monitor max T and liquid volume.
9. Export: liquid-region bounding box → W/D; probe T(t).

## Effort vs value

| Question | Use |
|----------|-----|
| Can commercial thermal FEM replace the twin for W/D? | **Tier A only** |
| Does continuum melt flow change W/D vs pure conduction? | Tier B without VOF |
| Do we need free-surface bead + deposition? | Twin or Fluent+VOF+UDFs (expensive) |

For the decision “is pursuing waam_twin necessary?”, finish **Tier A + macrograph** first.
Only open Tier B if Tier A fails for the right reasons (penetration physics), not bead cosmetics.
