# Second macrograph slot (held-out prediction)

The twin has **one** fitted experimental lock:

| Case | Process | Macrograph |
|------|---------|------------|
| `bead_calibrate.yaml` | 100 A × 15 V, 6.5 mm/s | W=7.0 × D=3.0 mm |

A second measured cut is required for absolute prediction claims. The slot is:

**`jobs/examples/bead_calibrate_heldout_macro2.yaml`** — same Goldak/η/recoil, travel **5.0 mm/s**.

## How to fill

1. Weld bead-on-plate at **100 A × 15 V, 5.0 mm/s** (same wire/gas/CTWD as calibrate if possible).
2. Measure pool width and penetration on the macrograph.
3. Edit the job:

```yaml
reference:
  awaiting_measurement: false
  pool_width_mm: <measured W>
  pool_depth_mm: <measured D>
  source: macrograph_ER70S-6_5mms_<date>
```

4. Report (no retuning):

```bash
PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.prediction_report
# or
WAAM_HELDOUT_VALIDATION=1 PYTHONPATH=. python3 -m waam_twin.validation.test_heldout_prediction
```

Absolute gate: max(W,D) error **&lt; 40%** (looser than the fitted 30% calibrate gate).

Until filled, the suite reports `PENDING` and only checks that the predicted pool is **larger** than calibrate (slower travel).

## Related

- Trend-only held-outs: `bead_calibrate_heldout_fast.yaml`, `bead_calibrate_heldout_hot.yaml`
- Write-up: [reference_case_ER70S6.md](reference_case_ER70S6.md)
