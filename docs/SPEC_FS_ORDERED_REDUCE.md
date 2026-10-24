# Spec: ordered reductions for a repeatable undamped bead

Status: proposed, 2026-10-23. Follows section 16 of
`docs/HANDOFF_FSLBM_CALIBRATION.md`.

## Why

Section 16 left two quiet 0.3 mm undamped welds (ceiling 1.486 m/s,
`mach_frac` ≈ 0.003) with mean W/D near Park and ΔW = 0.35 mm. The droplet
budget and the excess split already run in index order. What still writes a
schedule-dependent number back into the metal is an atomic sum:

- Unplaced excess in `fs_distribute_excess` uses `atomic_add` into
  `fs_mass_lost`; `fs_settle_unplaced` then spreads that share over every
  surface cell.
- Goldak with `normalize_energy=True` (shop default) divides every heated
  cell's ΔH by an f32 atomic weight sum in `inject_goldak_heat`.

This round replaces those sums with index-ordered reductions, proves the bead
state matches across two processes, and only then repeats the 0.3 mm undamped
weld. Do not edit the shop job.

## 1. Ordered sum

One f32 scratch volume on the grid, plus a small f64 partial buffer (block
length 256). Each block sums its cells in flat-index (C) order; a serialized
pass then adds the partials in block order. That total is the same bits as
summing `field.to_numpy().ravel(order="C")` in float64.

Do not serialize a walk of the whole grid. Do not serialize `fs_stream`.

## 2. Kernel wiring

- **Unplaced mass.** In `kernels/fslbm.py` `fs_distribute_excess`, a cell with
  `donor_cnt == 0` writes its excess into the scratch (zero elsewhere).
  Ordered-sum into `fs_mass_lost` before `fs_count_surface`. The integer
  surface count stays as it is.
- **Ledger.** The same kernel's `atomic_add` into `energy_acc[3]`, and the
  parallel `energy_acc[3] +=` in `fs_settle_unplaced` and `fs_apply_energy`,
  become per-cell scratch values summed in the same order and added once.
  This stops the ledger racing. It is not the acceptance test.
- **Goldak / Gaussian arc.** Write weights into the scratch, ordered-sum in
  f64, store that total once in `arc_norm_buf` before pass 2. Apply the same
  ordered sum to `inject_arc_heat`. Absolute-CF mode (`normalize_energy == 0`)
  does not rescale H; only its ledger sum needs the ordered path.

`fs_deposit_into_interface` and `feed_wire_surface` stay as they are.

## 3. Gates

1. Existing `test_fslbm` still passes. A new test fills a known excess pattern
   and checks the reducer against `numpy` in C-order, and checks two Goldak
   injects on a frozen `phi` leave the same `H` bits.
2. Two fresh processes, same command as the section 16 weld but
   `--time-s 0.5`, printing a sha256 of `phi`, `flags`, and `H`. Add that as
   a stop-early hash on `tools/steady_bead.py` so the job, dx, `C_darcy`, and
   `dt_scale` are the real weld. About 6 minutes each. If the hashes differ,
   stop. Do not start the 4 s pair.
3. Only if the hashes match: two full welds,
   `--time-s 4.0 --x0 22 --x1 34 --dx 0.3 --c-darcy 0 --dt-scale 0.5`,
   appended to `runs/steady/dx030_ordered.jsonl` (leave
   `dx030_undamped.jsonl` untouched). Abort if the printed ceiling is below
   1.4 m/s.

## 4. What to report

Append section 18 to `docs/HANDOFF_FSLBM_CALIBRATION.md` before "Known risks /
gotchas".

- **Hashes differ.** Quote both hashes and the step count. The ordered sums
  were not sufficient. No width claim.
- **Hashes match, pair inside the rule.** Both `mach_frac <= 0.05`, both
  steady, widths within 0.25 mm, mean W at least 5.80 mm, mean D within 10%
  of 1.8 mm. This mesh can host the undamped bead. Shop lock stays. Next
  spec is the HPC job at this dx with `C_darcy` 0 and this `dt_scale`, plus
  the 15 mm/s held-out.
- **Hashes match, still not repeatable or still narrow.** The numerical seed
  was not the 0.35 mm gap. Say so. Do not run 0.2 mm and do not open a
  damping screen in this spec.

## Out of scope

- Editing `bead_calibrate.yaml` or the shop held-outs.
- A non-zero `C_darcy`, a 0.2 mm weld, or raising `u_mach_limit_lu`.
- Serializing `fs_stream` or any other kernel that already writes only its own
  cell.
