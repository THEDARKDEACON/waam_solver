"""Fusion-zone mixing lerps alloy_frac; default path keeps birth composition."""

from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi, reset_taichi
from waam_twin import WAAMTwin
from waam_twin import kernels
from waam_twin.kernels import ALLOY_PLATE, ALLOY_WIRE
from waam_twin.materials import load_material


def run() -> float:
    reset_taichi()
    init_taichi(backend="cpu")
    wire = load_material("materials/placeholders/ER70S-6.yaml")
    plate = load_material("materials/placeholders/SS316L.yaml")
    twin = WAAMTwin(
        material=wire,
        plate_material=plate,
        nx=16, ny=12, nz=14, dx=8e-4, max_tracers=8,
        enable_alloy_mixing=True,
    )
    twin.alloy_mix_rate = 0.5
    twin.reset()
    g = twin.grid
    i0, i1, j0, j1 = twin.resolve_plate_ij()
    ip = (i0 + i1) // 2
    jp = (j0 + j1) // 2
    kp = max(1, twin.nz_solid - 2)

    flags = g.flags.to_numpy()
    fl = g.f_l.to_numpy()
    frac = g.alloy_frac.to_numpy()
    aid = g.alloy_id.to_numpy()
    # Two neighbouring liquid cells: plate and wire.
    flags[ip, jp, kp] = g.FLAG_FLUID
    flags[ip + 1, jp, kp] = g.FLAG_FLUID
    fl[ip, jp, kp] = 1.0
    fl[ip + 1, jp, kp] = 1.0
    frac[ip, jp, kp] = 1.0
    frac[ip + 1, jp, kp] = 0.0
    aid[ip, jp, kp] = ALLOY_PLATE
    aid[ip + 1, jp, kp] = ALLOY_WIRE
    g.flags.from_numpy(flags)
    g.f_l.from_numpy(fl)
    g.alloy_frac.from_numpy(frac)
    g.alloy_id.from_numpy(aid)

    for _ in range(40):
        kernels.mix_alloy_fusion_zone(
            g.alloy_frac, g.alloy_frac_buf, g.f_l, g.flags,
            twin.alloy_mix_rate, g.FLAG_GAS, g.nx, g.ny, g.nz,
        )
    out = g.alloy_frac.to_numpy()
    fp = float(out[ip, jp, kp])
    fw = float(out[ip + 1, jp, kp])
    birth = g.alloy_id.to_numpy()
    print(f"[alloy_mixing] frac plate-cell={fp:.3f} wire-cell={fw:.3f}")
    if abs(fp - 0.5) > 0.08 or abs(fw - 0.5) > 0.08:
        raise AssertionError(f"fusion mix did not approach 0.5: plate={fp:.3f} wire={fw:.3f}")
    if int(birth[ip, jp, kp]) != ALLOY_PLATE or int(birth[ip + 1, jp, kp]) != ALLOY_WIRE:
        raise AssertionError("mixing must not rewrite birth alloy_id")

    # Off path: melted plate keeps frac=1.
    reset_taichi()
    init_taichi(backend="cpu")
    off = WAAMTwin(
        material=wire, plate_material=plate,
        nx=16, ny=12, nz=14, dx=8e-4, max_tracers=8,
        enable_alloy_mixing=False,
    )
    off.reset()
    go = off.grid
    i0, i1, j0, j1 = off.resolve_plate_ij()
    ip = (i0 + i1) // 2
    jp = (j0 + j1) // 2
    kp = max(1, off.nz_solid - 2)
    if abs(float(go.alloy_frac.to_numpy()[ip, jp, kp]) - 1.0) > 1e-6:
        raise AssertionError("plate birth frac should be 1 with mixing off")
    return abs(fp - 0.5)


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
