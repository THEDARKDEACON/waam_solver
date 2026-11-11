# Server platform check and finer-grid mesh study — pre-registration
(written 12 Nov 2026, ~01:10, before any server run below; the one server smoke run
already done is listed as such)

## Why
The laptop (RTX 3050, 4 GB) cannot go below dx 0.3 mm in reasonable time or memory,
and the mesh study (docs/MESH_STUDY_PREREG.md) ended "not converged" on 0.5/0.4/0.3.
A shared server (2× RTX 4500 Ada, 24 GB; runs in the user's podman image with scipy
and scikit-learn added; Python 3.10, quadrants 1.3.3) can reach finer grids. Each
comparison below runs on one platform only.

Already done (not part of the tests): one smoke run, run04 M1 settings dx 0.5:
W 4.52, D 0.71, H_frozen 3.05, area ratio 1.04, mass conserved; 444 s (laptop 362 s).

## Common settings (= mesh stage M1 / V10 control C)
run04, η 0.819, b 2.42 mm, 3.5 s, window x 24–28 mm, `--u-design 2.0 --tau-min 0.5002`,
Smagorinsky C_s 0.17, viscosity ×1. Code snapshot 2ea629e (same solver code as the
laptop runs of C; later commits change docs only). One GPU (card 0). A run counts if it
has no NaN and window speed-cap hits ≤ 2 %.

## S1 — platform check (server vs laptop control C)
Server: dx 0.4 × 3 and dx 0.3 × 3. Laptop reference: control C (dx 0.4 and 0.3, n = 3
each). Per output (D, W, W_bead, H_frozen) and grid:
**equivalent** if |server mean − laptop mean| ≤ 2·SE (SE from both sets' repeat SDs)
or ≤ 6 %. All 8 equivalent → server and laptop results may be reported side by side.
Otherwise the platforms are kept separate and the difference is reported.

## S2 — timing (not a test)
Wall time per run, alone and with 3 runs sharing the card; peak GPU memory per grid.
Used only to choose how many runs to share a card.

## S3 — finer grids on the server
Server-only refinement sequence of the M1 model: dx 0.4, 0.3 (from S1), 0.2, and 0.15
if it fits in 24 GB and finishes in ≤ 8 h per run; up to 3 repeats each, as time allows
(fewer repeats are reported as such).
* Per output (D, W, W_bead, H_frozen), change between the two finest grids:
  **converged** if ≤ 6 % or within 2·SE (as the mesh study); for D also reported against
  the 10 % band of the V10 spec.
* Celik GCI reported on the three finest grids where the sequence is monotonic;
  oscillatory or divergent sequences reported as such.
* Width–power response is not tested here (run04 only).
Outcome rule, fixed now: if S3 converges at dx 0.2–0.15, the finest affordable grid
becomes a candidate for a single recalibration (a separate, later pre-registration);
if not, the model stays fixed-grid at dx 0.4 and the S3 numbers are reported.

## Politeness rules for the shared machine
Card 0 only; a batch starts only when the card shows < 1 GB used and < 10 %
utilisation; low CPU priority; time limits; only own files under ~/waam_twin_runs.

Results reported whatever they are.
