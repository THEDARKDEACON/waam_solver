# Energy / mass ledger

GPU-tracked conservation bookkeeping lives in `physics/ledger.py` and is
exposed in telemetry as `energy_ledger`.

## Terms (SI)

| Symbol | Meaning |
|--------|---------|
| `E_arc_J` | ∫ η·V·I dt (arc contribution after droplet split) |
| `E_droplet_J` | enthalpy booked for gas→fluid conversions |
| `E_conv_J` / `E_rad_J` | surface convection / radiation |
| `E_evap_J` | evaporative sink |
| `E_cap_J` / `E_floor_J` | enthalpy ceiling / floor corrections |
| `E_freeze_J` | trailing-freeze energy destroyed (should be ~0) |
| `E_stored_J` | ∫ H dV over metal |
| `m_wire_kg` | wire mass fed |
| `m_deposited_kg` | converted gas→fluid volume × ρ |
| `m_phi_above_kg` | φ-metal mass above the plate |

## Closure

```
residual = E_in − E_out − (E_stored − E_stored_0) + E_floor
rel = |residual| / max(|E_in|, |E_stored_0|, 1)
```

Target: `rel ≤ 2 %` on a short bead after H6 (enthalpy-cap init) is fixed.

## Tools

```
python -m waam_twin.tools.diagnostics.energy_ledger --job … --steps 2000
```

## Known defects closed

* **H6** — step-1 cap used uninitialised `cp_rho_field` → wiped plate energy.
  Fixed by `_refresh_thermal` before the first ceiling clamp.
* **H2** — deposition pass 3 double-counted droplet heat; removed.
* **H3** — trailing freeze discarded energy; disabled by default.
