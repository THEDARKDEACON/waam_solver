# LBM numerics (waam_twin v2)

## Scheme

- **Lattice:** D3Q19 velocity set with SRT (minimal/standard) or MRT (high/ultra presets).
- **Forcing:** Guo et al. (2002) body-force scheme for Marangoni, buoyancy, arc pressure, CSF.
- **Mushy zone:** Semi-implicit Carman–Kozeny drag on velocity before collision.
- **Streaming:** Pull scheme with bounce-back on solid; gas cells use pulled populations.

## Lattice units

| Quantity | Definition |
|----------|------------|
| Δx | Physical cell size [m] (`grid.dx`) |
| Δt | `dx · u_ref_lu / u_ref_phys` targeting Ma ≈ 0.05 |
| τ | `3 ν_lu + 0.5` where `ν_lu = (μ/ρ) Δt / Δx²` |
| Force | `F_lu = F_phys · Δt² / (ρ Δx)` |

## Variable properties (Phase 1–2)

When `use_material_tables=True`:

- `cp(T)`, `k(T)` → per-cell thermal diffusion (`alpha_lu_field`).
- `dγ/dT(T)` → Marangoni CSF scale (`dgamma_lu_field`).
- `μ(T)` → per-cell `tau_field` when `use_variable_tau=True` (SRT path).

MRT collision currently uses uniform `omega`; variable-τ MRT is a future stretch.

## Free surface

- VOF `φ` advected donor-cell; flags derived from φ.
- `surface_height_at(i,j)` sets arc injection height on the moving pool top.
- Balanced-force CSF: `F = γ κ ∇φ` with `κ = -∇·n̂`.

## Validation

| Test | Checks |
|------|--------|
| `test_lbm_poiseuille` | Parabolic profile preservation |
| `test_mass_conservation` | ρ drift < 2% |
| `test_laplace` | Non-zero CSF on interface |
| `test_soak_10k` | 10k steps, no NaN |

## References

- Guo et al., *Discrete lattice effects on the forcing term in the lattice Boltzmann method*, Phys. Rev. E 65 (2002).
- Krüger et al., *The Lattice Boltzmann Method* (Springer).
