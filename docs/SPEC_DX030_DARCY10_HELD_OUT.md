# Spec: held-out validation on the 0.3 mm / C_darcy 10 protocol

Status: proposed, 2026-10-24. Follows section 24 of
`docs/HANDOFF_FSLBM_CALIBRATION.md`.

## Why

Repeatability is closed at dx 0.3 mm, viscosity-floor step, and C_darcy 10
(`runs/steady/dx030_darcy10.jsonl`): mean W 5.82 mm, dW 0.07 mm, both steady.
Shop `bead_calibrate.yaml` stays C_darcy 160000 and dt_scale 1.0. C_darcy 10 is
a mesh/numerics lever, not a Park material fit.

Residuals vs Park 6.1 / 1.8 / 2.2 are about -5% / -8% / -11%. Fitting eta,
Goldak, or Marangoni to that one coupon before held-outs would overfit. This
spec freezes the protocol and scores held-out *trends* with no retune.

## 1. Frozen protocol

Tool overrides only (`tools/steady_bead.py`). Do not edit the shop job or the
held-out YAML locks.

| flag | value |
|------|-------|
| `--dx` | 0.3 |
| `--c-darcy` | 10 |
| `--dt-scale` | 0.5 |
| knobs | eta 0.82, Goldak 2.0/3.8/2.8/2.6, marangoni 1.0, contact 80, sulphur 28 |

No `--eta`, `--marangoni`, `--b`, `--c`, `--contact`, or `--arc-sigma` on
held-out runs. Abort if the printed ceiling is below 1.4 m/s.

**Calibrate baseline (already done):** `--job jobs/examples/bead_calibrate.yaml
--time-s 4.0 --x0 22 --x1 34`, pair in `dx030_darcy10.jsonl`. Pass: both
`steady=True`, dW <= 0.25 mm, mean W >= 5.80 mm, area_ratio near 1.

## 2. Held-out runs

Each case is one pair (`--repeat 2` or two fresh processes). Append to the
named jsonl under `runs/steady/`. Score trend and order of magnitude against
the calibrate baseline means (W 5.82, D 1.66, H_frozen 1.96). Do not retune.

| tag | job | window | expect |
|-----|-----|--------|--------|
| fast | `bead_calibrate_heldout_fast.yaml` (15 mm/s) | `--time-s 2.666 --x0 30 --x1 36` (path end at 40/15 s; past startup ramp) | W and D both below the 10 mm/s means |
| hot | `bead_calibrate_heldout_hot.yaml` (120 A) | `--time-s 4.0 --x0 22 --x1 34` | pool larger than 91 A (W or D or both up) |
| macro2 | `bead_calibrate_heldout_macro2.yaml` (overhead cite) | `--time-s 4.0 --x0 22 --x1 34` | soft only; position is not modeled. Cite is W 5.7 / D 1.9 / H 2.5 |

Common flags after the job and window:

```
--dx 0.3 --c-darcy 10 --dt-scale 0.5 --repeat 2
```

Example (fast):

```
python -m waam_twin.tools.steady_bead --job jobs/examples/bead_calibrate_heldout_fast.yaml --time-s 2.666 --x0 30 --x1 36 --dx 0.3 --c-darcy 10 --dt-scale 0.5 --repeat 2 --out runs/steady/dx030_darcy10_heldout_fast.jsonl
```

## 3. Decision rules

- **Trends hold.** Fast shrinks W and D relative to calibrate; hot grows the
  pool; each pair stays steady with dW <= 0.25 mm. Then a *one-knob* residual
  screen is allowed on the calibrate job only: eta *or* Goldak `c`, not both,
  not Marangoni in the same step. Stop if the move breaks dW or area_ratio.
- **Trends fail.** Record the bias in the handoff. Do not fit. Keep baseline
  knobs.
- **macro2** never gates a fit; report vs the overhead cite and move on.

## 4. Recording

Append a section to `docs/HANDOFF_FSLBM_CALIBRATION.md` before "Known risks /
gotchas" with the run tables and the decision. Shop YAML stays unchanged unless
a residual knob is locked by a later decision.

## Out of scope

- Promoting C_darcy 10 into the shop 0.4 mm lock.
- Bit-identical GPU hashes.
- Multi-knob Park optimizer.
- Ansys Tier A (thermal eta/Goldak check; separate playbook, does not block
  the held-out decision).
- Fluent melt-pool (Tier B).
