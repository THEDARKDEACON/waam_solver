# Mesh dependence and speed caps — short report (12 Nov 2026)

Model: run04 (Lambiase 2026, 2.9 kW, 750 mm/min), η 0.819, b 2.42 mm, C_s 0.17,
speed ceiling 2 m/s (`--u-design 2.0 --tau-min 0.5002`). Pre-registrations:
docs/MESH_STUDY_PREREG.md, docs/SERVER_MESH_PREREG.md, docs/CONDUCTION_DEPTH_PREREG.md.

## 1. Width and height are grid-independent; depth is not
Server, n = 3 per grid (mean ± SD, mm):

| output | dx 0.4 | dx 0.3 | dx 0.2 | 0.3 → 0.2 | verdict |
|---|---|---|---|---|---|
| Width W | 4.66 ± 0.06 | 4.39 ± 0.06 | 4.43 ± 0.06 | +0.8 % | converged |
| Bead width | 4.81 ± 0.08 | 4.44 ± 0.03 | 4.49 ± 0.07 | +1.0 % | converged |
| Frozen height | 3.20 ± 0.15 | 3.13 ± 0.08 | 3.16 ± 0.00 | +0.8 % | converged |
| Depth D | 0.725 ± 0.099 | 0.600 ± 0.056 | 0.530 ± 0.016 | −11.7 % | **not converged** |

* Depth falls monotonically with shrinking steps (Celik: order 2.7, extrapolated
  D ≈ 0.50 mm, GCI 8.2 % at dx 0.2), so it is approaching a limit but misses the
  6 % / 2·SE rule (|d| 0.070 vs 2·SE 0.067).
* Repeat scatter in depth falls 6× with refinement; the two-cluster depths at dx 0.4
  are a coarse-grid artefact.
* The calibrated dx 0.4 depth (≈ 0.72 mm) carries ≈ 30 % grid error. On a converged
  grid the model under-predicts penetration (Lambiase run04 measured 1.07 mm).
* Laptop and server agree (S1: 8/8 outputs equivalent).

## 2. Speed caps rise with refinement
A run counts only if the speed cap fires on ≤ 2 % of steps in the measuring window.

(run04 records from both platforms, laptop M1/C and server S1/S3.)

| dx (mm) | runs | window cap hits | counted |
|---|---|---|---|
| 0.5 | 2 | 0 % | 2/2 |
| 0.4 | 6 | 0 % | 6/6 |
| 0.3 | 7 | 0–12 % | 6/7 |
| 0.2 | 3 | 0–0.7 % | 3/3 |
| 0.15 | 2 (+1 running) | 7.7 %, 13.7 % | 0/2 |

The ceiling is a fixed 2 m/s, so the cap fires when finer grids resolve sharper
local speed peaks (droplet impact, thin surface films). At dx 0.15 it fires too
often to trust: **dx 0.2 is the finest usable grid at this ceiling.** Going finer
needs a higher ceiling, which shortens the time step (≈ 1.5× more cost at 3 m/s)
and changes the numerics, so it would need its own pre-registration.
The uncounted dx 0.15 values (D 0.59 / 0.51, W 4.29 / 4.56) agree with dx 0.2.

## 3. Where the depth drift comes from (running)
Conduction only (no flow, no wire; absolute depth not comparable, trend only):
dx 0.5 → D 2.11 mm, dx 0.4 → 2.19 mm (+4 %); dx 0.3 / 0.25 / 0.2 running.
If depth stays flat here, the drift lies in the flow / free surface / deposit,
not in the heat source or the depth measurement.

## Verdict (pre-registered rule)
S3 does not converge on depth at the finest counted grids (0.3 → 0.2), so the model
**stays fixed-grid at dx 0.4**: width and height results stand; depth agreement is
partly grid error. Pending: the third dx 0.15 run and the conduction check.

## Next
1. Finish the conduction check → locate the depth drift.
2. If the third dx 0.15 run counts, re-apply the rule on 0.2 → 0.15.
3. Option: pre-register one recalibration at dx 0.2 (≈ 2 h/run on the server),
   stating depth with its 8 % grid uncertainty.
