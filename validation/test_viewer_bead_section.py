"""
test_viewer_bead_section.py — HUD bead size from one cross-section.

Synthetic sections at dx 0.4 mm: a 6 mm × 2 mm lens bead, the same bead
with a spatter blob beside it (ignored), a three-layer wall (full height),
a liquid bead (φ < 1 at the top still counts), and an empty plane.
"""

from __future__ import annotations

import numpy as np

from waam_twin.viewer.bead_section import measure

DX = 0.4
NT, NZ, KF = 60, 40, 15  # columns, layers, plate face


def _lens(sec: np.ndarray, t_c: float, half_w_mm: float, h_mm: float, k0: int = KF) -> None:
    """Parabolic lens, sub-cell φ in the top cell of each column."""
    for t in range(NT):
        y = (t - t_c) * DX
        if abs(y) >= half_w_mm:
            continue
        h = h_mm * (1.0 - (y / half_w_mm) ** 2)
        full = int(h / DX)
        sec[t, k0:k0 + full] = 1.0
        sec[t, k0 + full] = (h - full * DX) / DX


def _plate() -> np.ndarray:
    sec = np.zeros((NT, NZ), np.float32)
    sec[:, :KF] = 1.0
    return sec


def run() -> None:
    sec = _plate()
    _lens(sec, 30, 3.0, 2.0)
    h, w, toe = measure(sec, KF, 30, DX)
    print(f"[bead_section] lens: h={h:.2f} w={w:.2f} toe={toe:.0f}°")
    assert abs(h - 2.0) < 0.05, h
    assert abs(w - 6.0) < 2 * DX, w
    # Lens edge slope 2h/half_w = 1.33 → 53°; a one-cell reading is coarse.
    assert 30.0 < toe < 70.0, toe

    spat = sec.copy()
    spat[45:47, KF:KF + 3] = 1.0  # spatter 6 mm off the centreline, not joined
    h2, w2, _ = measure(spat, KF, 30, DX)
    assert abs(h2 - h) < 1e-9 and abs(w2 - w) < 1e-9, (h2, w2)

    wall = _plate()
    for layer in range(3):
        _lens(wall, 30, 3.0, 2.0, k0=KF + 5 * layer)
        wall[24:37, KF + 5 * layer:KF + 5 * layer + 3] = 1.0  # fused layer core
    h3, w3, _ = measure(wall, KF, 30, DX)
    print(f"[bead_section] wall: h={h3:.2f} w={w3:.2f}")
    assert h3 > 3 * 1.8, h3

    liq = _plate()
    _lens(liq, 30, 3.0, 2.0)
    liq[:, KF + 4] *= 0.5  # liquid top cells half full
    h4, _, _ = measure(liq, KF, 31, DX)
    assert 1.7 < h4 <= 2.0, h4

    assert measure(_plate(), KF, 30, DX) == (0.0, 0.0, 0.0)
    off = _plate()
    _lens(off, 10, 2.0, 1.5)  # bead far from the given centreline
    assert measure(off, KF, 30, DX) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    run()
    print("PASS")
