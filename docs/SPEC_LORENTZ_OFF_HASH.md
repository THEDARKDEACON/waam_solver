# Spec: Lorentz-off hash gate

Status: proposed, 2026-10-23. Follows section 18 of
`docs/HANDOFF_FSLBM_CALIBRATION.md`.

## Why

Section 18 ordered the unplaced-mass sum and the Goldak weight sum. Two
processes at 0.5 s still hashed `phi`, `flags`, and `H` differently (30,952
steps, Mach hits 24 versus 31). `--lorentz-every-s` is the wrong switch:
`lorentz_due` still solves when `_step_n % n == 0`, so step 0 always runs and
the cached force is reapplied. This round turns Lorentz off for real and
repeats that hash. It does not rewrite a kernel. Do not edit the shop job.

## 1. Switch

Add `--no-lorentz` to `tools/steady_bead.py`. In `_apply_twin_overrides`, set
`twin.enable_lorentz = False`. Record `enable_lorentz` on the JSON record.
Leave `jobs/examples/bead_calibrate.yaml` unchanged.

## 2. Gate

Two fresh processes, same weld as section 18, with Lorentz off. About 8 minutes
each. Abort if the printed ceiling is below 1.4 m/s.

```
python -m waam_twin.tools.steady_bead --job jobs/examples/bead_calibrate.yaml --time-s 0.5 --x0 22 --x1 34 --dx 0.3 --c-darcy 0 --dt-scale 0.5 --no-lorentz --hash-fields --out runs/steady/dx030_nolorentz_hash.jsonl
```

Run it twice so the file holds both hashes. Compare sha256 of `phi`, `flags`,
and `H`, and record the Mach hit counts.

## 3. What to report

Append section 20 to `docs/HANDOFF_FSLBM_CALIBRATION.md` before "Known risks /
gotchas".

- **Hashes match.** Lorentz is required for the split seen in section 18. The
  live write-back is `elec_bin_axial_current` in `kernels/lorentz.py`, which
  atomic-adds `Jz·dA` into the radial bins that become the magnetic force.
  The next spec orders that sum the same way as the Goldak weight, then
  repeats the 0.5 s hash with Lorentz on. Leave the Jacobi residual atomic
  until that hash has been run. No 4 s weld in this spec.
- **Hashes differ.** Lorentz is not required for the split. The next spec is
  the droplet budget on CUDA across two processes. The identical-fill check
  in `test_fslbm` only passed on the CPU inside one process. No Lorentz-bin
  rewrite and no 4 s weld in this spec.

## Out of scope

- Editing `bead_calibrate.yaml` or the shop held-outs.
- Ordering `elec_bin_axial_current`, changing the Jacobi residual, or
  replacing the droplet reserve.
- A 4 s weld, a 0.2 mm weld, a non-zero `C_darcy`, or raising
  `u_mach_limit_lu`.
