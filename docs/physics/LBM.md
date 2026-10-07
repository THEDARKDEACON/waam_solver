# LBM numerics (waam_twin v2)

## Scheme

- **Lattice:** D3Q19. Collision is chosen by job `use_srt` (not by preset alone): SRT or two-rate central-moment MRT. Presets supply a default `use_srt` that jobs override and lock.
- **Forcing:** Guo et al. (2002) body-force scheme for Marangoni, buoyancy, arc pressure, CSF. Variable-τ MRT uses regenerate-and-rediff \(S=0.5\,F_i\) plus an \(f_0\) mass correction (see `cumulant_kernel.py`, `test_mrt_force_mass`).
- **Mushy zone:** Semi-implicit Carman–Kozeny drag inside collide. Job `C_darcy` is defined at `C_darcy_ref_dt_s`; kernels use \(C_{\mathrm{eff}}=C\cdot(\Delta t/\Delta t_{\mathrm{ref}})\).
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
- `μ(T)` → per-cell `tau_field` when `use_variable_tau=True` (SRT and two-rate MRT paths).

Variable-τ two-rate MRT is implemented in `cumulant_kernel.collide_mrt_variable_tau` (wired from `kernels/lbm.py`). Mach clamp resyncs `f` via equilibrium projection (`test_mach_clamp_population_sync`). Enthalpy-ceiling discard and φ metal volume above substrate are ledgered in telemetry.

## Free surface

- VOF `φ` advected donor-cell; flags derived from φ.
- `surface_height_at(i,j)` sets arc injection height on the moving pool top.
- Balanced-force CSF: `F = γ κ ∇φ` with `κ = -∇·n̂`.

## Validation

| Test | Checks |
|------|--------|
| `test_lbm_poiseuille` | Parabolic profile preservation |
| `test_mass_conservation` | ρ drift < 2% |
| `test_laplace` | CSF Laplace pressure / interface force gate (mass conservation is separate) |
| `test_mrt_force_mass` | Two-rate MRT mass + Guo force impulse vs SRT |
| `test_remelt_population_init` | SOLID→FLUID reinit of equilibrium populations |
| `test_mach_clamp_population_sync` | Mach clamp resyncs population momentum to capped `u` |
| `test_operator_splitting` | Peak-T change between `dt` and `dt/2` |
| `test_soak_10k` | 10k steps, no NaN |

## References

- Guo et al., *Discrete lattice effects on the forcing term in the lattice Boltzmann method*, Phys. Rev. E 65 (2002).
- Krüger et al., *The Lattice Boltzmann Method* (Springer).
