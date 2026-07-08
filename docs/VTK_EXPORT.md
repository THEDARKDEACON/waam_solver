# VTK export — field reference

This document lists every array written to VTK files by `waam_twin`, how each quantity is computed in the simulation, and how to export them.

**Implementation:** [`export/vtk_io.py`](../export/vtk_io.py), [`export/bundle.py`](../export/bundle.py), [`export/meta.py`](../export/meta.py).

---

## Export entry points

| API | Output | Default tiers |
|-----|--------|---------------|
| `twin.export_vtk(path)` | `.vti` volume | Tier 0 + 3 |
| `twin.export_vtk_full(path, tiers=(0,1,2,3))` | `.vti` volume | All tiers |
| `twin.export_haz_vtk(path)` | `.vti` volume | Tier 0 (full core set, not HAZ-only) |
| `twin.export_surface_vtk(path)` | `.vtp` surface mesh | φ or f_l = 0.5 contour |
| `twin.export_tracers_vtk(path)` | `.vtp` point cloud | Porosity tracers |
| `twin.export_research_bundle(out_dir/)` | volume + surface + tracers + JSON | 0, 1, 3 |
| `python3 -m waam_twin.export ...` | Time series + `sequence.pvd` | CLI wrapper |

**Requirements:** PyVista installed (`pip install pyvista`). Set `WAAM_HEADLESS=1` to skip VTK in batch runs.

**Coordinates:** Volume files use **millimetres**. Origin X includes the moving-window offset (`window_offset_x_mm`). Spacing = `dx_mm` in all three axes.

**Storage layout:** Cell-centred arrays, Fortran order (`order="F"`), on PyVista `ImageData` with dimensions `(nx+1, ny+1, nz+1)`.

---

## Tier system

Tiers control which arrays are included in volume export (`export_vtk_full`):

| Tier | Name | When included |
|------|------|---------------|
| **0** | Core | Always (thermal, VOF, velocity, HAZ trackers) |
| **1** | Material | Only if `twin.use_material_tables` (T-dependent cp, k, dγ/dT, τ) |
| **2** | Forces | Body-force snapshot; Lorentz **J**, **B** if `enable_lorentz` |
| **3** | Derived | Computed at export time (curvature, vorticity, mushy mask) |

---

## Tier 0 — Core volume fields (`.vti`)

These are **live GPU fields** on `WAAMGrid`, updated each LBM timestep in [`solvers/coupled_step.py`](../solvers/coupled_step.py).

| VTK array | Grid field | Units | How it is computed |
|-----------|------------|-------|-------------------|
| `Temperature_K` | `T` | K | Recovered from enthalpy `H` each step via enthalpy–porosity (`kernels.update_phase` or `update_phase_variable_cp`). Mushy zone: linear T between T_solidus and T_liquidus. |
| `Temperature_C` | derived | °C | `T − 273.15` at export |
| `Liquid_Fraction` | `f_l` | 0–1 | From `H`: 0 in solid, 1 in liquid, linear in mushy zone between H_sol and H_liq. |
| `VOF_phi` | `phi` | 0–1 | Volume-of-fluid metal fraction (0 = gas, 1 = bulk metal). Advected with LBM velocity; reinitialized each step when `enable_vof`. |
| `Cell_Flags` | `flags` | bitmask | `0` fluid, `1` solid (substrate/walls), `2` gas, `4` interface (VOF). Updated from φ when VOF is on. |
| `T_max_K` | `T_max` | K | Running maximum of `T` per cell (HAZ envelope). Updated after thermal step; gas cells skipped. [`kernels.update_T_max`] |
| `T_prev_K` | `T_prev` | K | Temperature at the **previous** timestep (used internally for cooling rate). |
| `Cooling_Rate_Ks` | `dT_dt` | K/s | `(T − T_prev) / dt` each step; then `T_prev ← T`. Gas cells skipped. [`kernels.update_cooling_rate`] |
| `Enthalpy_Jm3` | `H` | J/m³ | Primary thermal conserved variable. Advected/diffused; incremented by arc heat, droplet deposition, boundary losses. |
| `Time_above_800C_s` | `time_above_800_s` | s | Cumulative time `T ≥ 1073.15 K` (800 °C). [`kernels.update_time_above_T`] |
| `Time_above_1100C_s` | `time_above_1100_s` | s | Cumulative time `T ≥ 1373.15 K` (1100 °C). |
| `Time_above_solidus_s` | `time_above_solidus_s` | s | Cumulative time `T ≥ T_solidus` (material). |
| `Velocity_X_ms` | `ux` | m/s | LBM lattice velocity → physical: `u_ms = u_lu × dx / dt`. |
| `Velocity_Y_ms` | `uy` | m/s | Same conversion. |
| `Velocity_Z_ms` | `uz` | m/s | Same conversion. |
| `Speed_ms` | derived | m/s | `√(ux² + uy² + uz²)` in m/s at export. |
| `Density_lu` | `rho` | lu | LBM density from distribution moments (≈ 1 in fluid). |
| `Pressure_Pa` | derived | Pa | Ideal-gas estimate from LBM density: `P = ρ_lu × cs² × ρ_phys × (dx/dt)²`, `cs² = 1/3`. Illustrative, not a full compressible solver. |

### Thermal / phase pipeline (where Tier 0 fields come from)

Each `twin.step()` roughly:

1. Arc + droplet heating → update `H`
2. Advect/diffuse `H` → recover `T`, `f_l`
3. Boundary convection/radiation on `H`
4. `update_T_max`, `update_cooling_rate`, `update_time_above_T`
5. VOF: advect `phi`, contact-angle BC, update `flags`
6. LBM collide/stream → `ux`, `uy`, `uz`, `rho`

---

## Tier 1 — Material tables (`.vti`)

Included only when the job uses T-dependent material tables (`use_material_tables`, typical on `standard` preset with validated alloys).

| VTK array | Grid field | Units | How it is computed |
|-----------|------------|-------|-------------------|
| `RhoCp_Jm3K` | `cp_rho_field` | J/(m³·K) | `ρ × cp(T)` from material YAML tables, updated from local `T` each step. |
| `Alpha_lu` | `alpha_lu_field` | lu | Thermal diffusivity mapped to lattice units for the enthalpy solver. |
| `DgammaDT_lu` | `dgamma_lu_field` | lu | dγ/dT for Marangoni, in lattice units. |
| `Tau_SRT` | `tau_field` | – | Local SRT relaxation time from μ(T) (Carman–Kozeny mushy damping included in collide). |

---

## Tier 2 — Forces (`.vti`)

Body forces are **snapshotted at end of step** into `Fx_snap`, `Fy_snap`, `Fz_snap` so export sees the accumulated forcing for that timestep (before the next step clears `Fx`…`Fz`).

| VTK array | Source | Units | How it is computed |
|-----------|--------|-------|-------------------|
| `BodyForce_X_ms2` | `Fx_snap` | m/s² | `a = F_lu × dx / dt²` (same as LBM body-force → acceleration). |
| `BodyForce_Y_ms2` | `Fy_snap` | m/s² | Same. |
| `BodyForce_Z_ms2` | `Fz_snap` | m/s² | Same. |
| `BodyForce_mag_ms2` | derived | m/s² | Magnitude of the above. |

**Contributors to `Fx,Fy,Fz` during the step** (when respective flags are on):

- CSF surface tension (+ wetting wall term if `enable_wetting`)
- Marangoni shear from ∇T
- Buoyancy (Boussinesq)
- Hydrostatic gravity on liquid (`enable_hydrostatic_gravity`)
- Lorentz **J×B** (`enable_lorentz`)
- Gas-jet shear (`enable_gas_shear`)
- Arc pressure + droplet impact on free surface
- Vapor recoil (`enable_recoil`)

Lorentz fields (only if `enable_lorentz`):

| VTK array | Grid field | Units | How it is computed |
|-----------|------------|-------|-------------------|
| `CurrentDensity_X_Am2` | `Jx` | A/m² | **J** = −σ∇φ_e from solved electric potential in the weld pool [`kernels.elec_compute_J`] |
| `CurrentDensity_Y_Am2` | `Jy` | A/m² | Same. |
| `CurrentDensity_Z_Am2` | `Jz` | A/m² | Same. |
| `MagneticField_X_T` | `Bx` | T | **B** = μ₀ ∇×**J** (central differences) [`kernels.elec_compute_B_from_J`] |
| `MagneticField_Y_T` | `By` | T | Same. |
| `MagneticField_Z_T` | `Bz` | T | Same. |

---

## Tier 3 — Derived at export (`.vti`)

Computed in Taichi/NumPy when Tier 3 is requested; not stored on GPU between steps.

| VTK array | Units | How it is computed |
|-----------|-------|-------------------|
| `Curvature_kappa` | 1/m (cell units) | κ = −∇·n̂ from φ gradients; same convention as CSF kernel [`kernels.compute_curvature_field`]. Zero in gas cells. |
| `Vorticity_mag_1s` | 1/s | \|∇×**u**\| from lattice velocities, scaled to physical units [`kernels.compute_vorticity_magnitude`]. |
| `Mushy_Zone` | 0/1 | `1` where `0.05 < f_l < 0.95`, else `0`. |

---

## Surface export (`.vtp`)

`export_surface_vtk` builds a **PolyData** isosurface:

1. Sample `phi`, `Liquid_Fraction`, `Temperature_K`, optional `Curvature_kappa` on the volume grid.
2. Convert cell data → point data.
3. Contour at **0.5** on `Liquid_Fraction` first; if empty, try `phi = 0.5`.
4. Save the resulting mesh (bead crown / pool boundary).

Requires `enable_vof` and a visible metal–gas interface. Empty surface → export skipped with a log message.

---

## Tracer export (`.vtp`)

Porosity / inclusion tracers from the LBM tracer model:

| Point array | Meaning |
|-------------|---------|
| `state` | `porosity_active`: 0 inactive, 1 active, 2 trapped |
| `trapped` | `1` if trapped, else `0` |

Positions: `porosity_pos` in **mm**, with moving-window X offset applied.

---

## Research bundle (non-VTK sidecars)

`export_research_bundle` also writes:

| File | Contents |
|------|----------|
| `telemetry_step_*.json` | Scalar diagnostics from `twin.get_telemetry()` (pool W/D, T_peak, bead height, mass, etc.) |
| `meta_step_*.json` | Grid, material, physics flags, unit conversion notes, process snapshot |
| `probes.csv` | Time series if job defines `probes:` |

See [`export/meta.py`](../export/meta.py) for the full meta schema.

---

## Unit conversions (summary)

| Quantity | Lattice → SI |
|----------|--------------|
| Velocity | `u_ms = u_lu × dx / dt` |
| Acceleration (body force) | `a_ms2 = F_lu × dx / dt²` |
| Pressure (derived) | `P = ρ_lu × (1/3) × ρ_phys × (dx/dt)²` |

Recorded in `meta_*.json` → `unit_conversions`.

---

## Quick usage

```python
# Full field set
twin.export_vtk_full("pool.vti", tiers=(0, 1, 2, 3))

# Bead surface mesh
twin.export_surface_vtk("bead.vtp")

# Everything + metadata
twin.export_research_bundle("run_out/")
```

```bash
# Animated sequence for ParaView
python3 -m waam_twin.export \
  --job jobs/examples/bead_on_plate.yaml \
  --preset standard --steps 5000 --every 100 --out viewer_output/run1
# Open viewer_output/run1/sequence.pvd in ParaView
```

**ParaView tips**

- Open `sequence.pvd` for time series, not a single `.vti`.
- Threshold `Liquid_Fraction > 0.05` or `Cell_Flags == 0` to isolate melt.
- Contour `T_max_K` at 1073, 1373, or T_solidus for HAZ isotherms.
- `crop_liquid=True` on `export_vtk_full` thresholds to cells with `f_l > 0.05`.

---

## Related docs

- [README.md § VTK export](../README.md) — quick start commands
- [README.md § Interactive viewer](../README.md) — live GGUI keys and export shortcuts
