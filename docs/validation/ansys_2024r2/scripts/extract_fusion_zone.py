#!/usr/bin/env python3
"""Extract fusion-zone W×D from an Ansys nodal temperature CSV export.

Expected CSV columns (header required), case-insensitive:
  x_mm, y_mm, z_mm, T_C
or
  x, y, z, T   (interpreted as mm and °C if --units mm_C)

Fusion criterion: T >= T_liquidus_C from params JSON (default 1519.85 °C).

Width = max X extent of fused nodes near the hottest Y slice (travel is +Y).
Depth = max (z_top - z) among fused nodes on centerline band.

Usage:
  python3 extract_fusion_zone.py --csv ansys_T.csv \\
    --params ../params_bead_on_plate.json --out ../comparison/ansys_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _colmap(headers: list[str]) -> dict[str, int]:
    h = [c.strip().lower() for c in headers]
    aliases = {
        "x": ("x_mm", "x", "x [mm]", "x(mm)"),
        "y": ("y_mm", "y", "y [mm]", "y(mm)"),
        "z": ("z_mm", "z", "z [mm]", "z(mm)"),
        "t": ("t_c", "t", "temp", "temperature", "temperature_c", "temp [c]"),
    }
    out: dict[str, int] = {}
    for key, names in aliases.items():
        for n in names:
            if n in h:
                out[key] = h.index(n)
                break
    missing = [k for k in ("x", "y", "z", "t") if k not in out]
    if missing:
        raise SystemExit(f"CSV missing columns {missing}; got headers={headers}")
    return out


def load_nodes(path: Path) -> list[tuple[float, float, float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        idx = _colmap(headers)
        rows = []
        for row in reader:
            if not row or len(row) <= max(idx.values()):
                continue
            rows.append(
                (
                    float(row[idx["x"]]),
                    float(row[idx["y"]]),
                    float(row[idx["z"]]),
                    float(row[idx["t"]]),
                )
            )
    if not rows:
        raise SystemExit(f"No data rows in {path}")
    return rows


def fusion_metrics(
    nodes: list[tuple[float, float, float, float]],
    *,
    T_liq_C: float,
    x_center_mm: float,
    z_top_mm: float,
    center_band_mm: float = 1.0,
) -> dict:
    fused = [n for n in nodes if n[3] >= T_liq_C]
    if not fused:
        return {
            "pool_width_mm": 0.0,
            "pool_depth_mm": 0.0,
            "n_fused_nodes": 0,
            "peak_temp_C": max(n[3] for n in nodes),
            "note": f"No nodes with T >= {T_liq_C} C",
        }

    # Hottest Y among fused → representative cross-section
    y_hot = max(fused, key=lambda n: n[3])[1]
    slice_nodes = [n for n in fused if abs(n[1] - y_hot) <= center_band_mm]
    if len(slice_nodes) < 3:
        slice_nodes = fused

    xs = [n[0] for n in slice_nodes]
    width = max(xs) - min(xs)

    centerline = [
        n for n in fused if abs(n[0] - x_center_mm) <= center_band_mm
    ]
    if not centerline:
        centerline = fused
    depth = max(z_top_mm - n[2] for n in centerline)
    depth = max(0.0, depth)

    return {
        "pool_width_mm": round(width, 3),
        "pool_depth_mm": round(depth, 3),
        "n_fused_nodes": len(fused),
        "y_hot_mm": round(y_hot, 3),
        "peak_temp_C": round(max(n[3] for n in nodes), 2),
        "T_liq_C": T_liq_C,
        "definition": "T >= T_liquidus isotherm envelope",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time_s", type=float, default=2.0)
    args = ap.parse_args()

    params = json.loads(args.params.read_text(encoding="utf-8"))
    mat = params["material_ER70S6_constants"]
    geo = params["geometry_mm"]
    nodes = load_nodes(args.csv)
    metrics = fusion_metrics(
        nodes,
        T_liq_C=float(mat["T_liquidus_C"]),
        x_center_mm=float(geo["torch_path_start_xy"][0]),
        z_top_mm=float(geo["plate_thickness"]),
    )
    out = {
        "code": "ansys_mechanical_2024r2",
        "tier": "A_thermal_goldak",
        "source_job": params.get("source_job"),
        "time_s": args.time_s,
        "process": params["process"],
        "macro": metrics,
        "micro": {
            "note": "Fill probe peak / t85 / time_above from Mechanical charts",
            "probe_peak_temp_C": None,
            "t85_s": None,
            "time_above_800C_s": None,
            "time_above_1100C_s": None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        f"  W={metrics['pool_width_mm']} mm  D={metrics['pool_depth_mm']} mm  "
        f"peakT={metrics['peak_temp_C']} C  fused_nodes={metrics['n_fused_nodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
