# Ansys vs waam_twin comparison (Tier A)

- Ansys: comparison/ansys_metrics.json (max-over-time T >= T_liq fusion, Y=30)
- Twin: comparison/twin_metrics.json (bead_calibrate_thermal.yaml, physics_tier thermal, dx=0.4 mm, 1370 path steps, T_max fusion envelope)
- Tolerance band: +/-15% Ansys <-> twin on W and D
- Experiment: Park flat **6.1 x 1.8 mm** (DOI 10.3390/app9214626)

| Metric | Ansys | Twin (thermal) | Experiment | Ansys vs Twin | Ansys vs Exp | Twin vs Exp |
|--------|------:|---------------:|-----------:|--------------:|-------------:|------------:|
| Pool width mm | 3.262 | 3.400 | 6.1 | 4.0% | -46.5% | -44.3% |
| Pool depth mm | 1.601 | 1.633 | 1.8 | 1.9% | -11.1% | -9.3% |
| Peak T C | 2802.2 | 2760 | — | | | |

## Verdict (decision_rubric.md)

**Ansys <-> twin-thermal:** agree within +/-15% on W (4.0%) and D (1.9%). Same question answered: max-OT / T_max liquidus fusion envelope on Park-lock Goldak conduction.

**Vs Park experiment:**
- Both codes under-predict width (~-45%); depth is closer (~-10%).
- Ansys W remains short of Park; aligning the twin metric does not close that gap.

**Rubric row:** **(4)** — Ansys ≈ twin, both miss experiment the same way on W. Shared Goldak/thermal setup is the bottleneck for Park width; do **not** retune η/Goldak from Ansys alone. **Continue waam_twin** for geometry / free-surface / held-outs (Tier A cannot score crown). Prior full-CFD live-pool export (4.80×1.20) was a different metric and is retired for this Tier A compare.

## Suggested report sentence

On the Park 2019 flat P-GMAW window (91 A × 22 V, η=0.82, 10 mm/s; DOI 10.3390/app9214626), Ansys 2024 R2 Transient Thermal Goldak (max-over-time T≥T_liq) gave W×D = 3.26×1.60 mm; waam_twin thermal (bead_calibrate_thermal, measure_fusion_zone_mm) gave 3.40×1.63 mm; macrograph 6.1×1.8 mm. Relative Ansys–twin errors were 4.0% (W) and 1.9% (D). Decision: codes agree on Tier A; both miss Park W — continue twin for multiphysics geometry, not for matching Ansys by retuning Goldak.

## Notes

- Ansys: flat plate, no crown; Park-lock Q=1641.64 W, b=2.8, c=2.6.
- Twin primary fields are fusion envelope; live liquid pool kept as pool_width_live_mm / pool_depth_live_mm.
- Export drives jobs/paths/bead_line.csv (+Y, 20 mm / 2 s) via TorchPathDriver; steps derived from path length.
- Shop 0.4 mm lock unchanged; CFD working point remains dx 0.3 / C_darcy=10.
