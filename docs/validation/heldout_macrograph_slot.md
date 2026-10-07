# Published held-out macro slot (Park overhead)

The twin calibrate lock is the **Park et al. 2019 flat** fusion macro:

| Case | Process | Macrograph | Citation |
|------|---------|------------|----------|
| `bead_calibrate.yaml` | ≈91 A mean, 10 mm/s, WFR 7 m/min | W=6.1 × D=1.8 mm | [DOI 10.3390/app9214626](https://doi.org/10.3390/app9214626) |

**`jobs/examples/bead_calibrate_heldout_macro2.yaml`** carries the **overhead** cut from the same paper (W=5.7 × D=1.9 mm). Electrical process matches the flat lock; the experiment differed by **welding position**, which the twin does **not** model — treat as a soft absolute check.

Trend-only process variants (no absolute macro):

- `bead_calibrate_heldout_fast.yaml` — faster travel  
- `bead_calibrate_heldout_hot.yaml` — higher I/WFS  

Other grounded externals: Bruno (Figshare surface), PIONEER M1 (Zenodo wall). Catalog: [data/published_macrograph_catalog.json](data/published_macrograph_catalog.json).

```bash
PYTHONPATH=. WAAM_BACKEND=cuda python3 -m waam_twin.tools.prediction_report --tier shop --with-bruno
```
