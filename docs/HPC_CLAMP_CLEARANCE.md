# HPC clamp clearance (step 2)

After the TRT mass fix on the laptop, clear Mach/force clamps on a GPU that
can hold **dx ≲ 0.11 mm** at a production-sized plate.

## Why HPC

With τ ≥ 0.5005 and Ma ≤ 0.08, the acoustic dt bound is

\[
d_x \le \frac{3\nu\,\mathrm{Ma}}{u\,(\tau_{\min}-1/2)}
\approx 3.2\times10^{-4}/u
\]

For Marangoni \(u\sim 2\)–\(3\,\mathrm{m/s}\) that is **dx ≲ 0.16–0.11 mm**.
A 4 GB laptop OOMs or is too slow; use ≥16 GB VRAM (prefer 24–40 GB).

Do **not** raise `u_mach_limit_lu` or `force_limit_m_s2` to pass.

## Job knobs

Start from `jobs/examples/bead_clamp_clear.yaml` (or copy to
`jobs/examples/bead_clamp_hpc.yaml`) and set:

| Knob | Suggested |
|------|-----------|
| `simulation.dx_mm` | `0.10` (or `0.12` first smoke) |
| `simulation.domain_mm` | plate + ~10 mm halo (e.g. `[60, 30, 20]`) |
| `simulation.use_srt` | `false` (TRT) |
| `simulation.auto_dt_ma` | `true` |
| `simulation.u_design_m_s` | `2.5`–`3.0` (≥ expected max \|u\|) |
| `simulation.u_mach_limit_lu` | `0.08` |
| `simulation.force_limit_m_s2` | `12500` (unchanged) |
| `simulation.warn_on_force_clamp` | `false` (log spam) |
| `plate.size_mm` | real coupon |
| torch CSV | path inside the plate footprint |

VRAM ballpark: cells ≈ (Lx/dx)·(Ly/dx)·(Lz/dx); ~0.3–0.5 µB/cell for full Tier-F → 0.10 mm on 60×30×20 mm ≈ 36 M cells ≈ **10–15 GB**.

## Commands

From `waam_twin/` on the HPC node (module load CUDA as required):

```bash
export PYTHONPATH=..
export WAAM_BACKEND=cuda

# 1) Short equal-travel clamp probe (must exit 0)
python -m waam_twin.tools.diagnostics.clamp_check \
  --job jobs/examples/bead_clamp_hpc.yaml \
  --travel-mm 2.0
# Expect: "clamps_clear": true, force_clamp_steps=0, mach_clamp_steps=0,
#         peak_temp_C ≫ 1000, n_liquid > 0, no NaNs

# 2) Equal-travel GCI (Tier-F series around the clear dx)
WAAM_TRAVEL_MM=3.2 python -m waam_twin.tools.mesh_convergence_report \
  --job jobs/examples/bead_clamp_hpc.yaml \
  --dx 0.15,0.12,0.10 \
  --json validation/baselines/mesh_convergence_hpc_tierf.json
# Expect: verdict status PASS_GCI (or band gate), clamps_ok true
```

Copy JSON artifacts back to the laptop repo under `validation/baselines/`.

## If clamps still fire

1. Confirm TRT is active: log line `Collision: TRT (two-rate, Λ=1/4)`.
2. If **Mach** only: raise `u_design_m_s` or drop `dx_mm` (keep Ma=0.08).
3. If **force** only: check Lorentz Jacobi convergence (`lorentz_jacobi_iters`);
   CSF spikes at coarse dx often shrink when dx≤0.10 — do not raise the cap.
4. Re-check `choose_dt` feasibility: τ floor and Ma must both be satisfiable
   at the chosen (dx, u_design).

## Laptop smoke after reboot

Once CUDA works again locally:

```bash
PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.diagnostics.clamp_check \
  --job jobs/examples/bead_calibrate.yaml --travel-mm 2.0
```

Shop dx=0.4 will still FAIL_CLAMPS (expected). Use it only to confirm TRT no
longer NaNs through melt (`peak_temp_C` hot, finite ρ).
