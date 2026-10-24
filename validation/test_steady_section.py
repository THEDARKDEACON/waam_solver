"""Host-side checks for the frozen-bead height metric. No GPU."""

from __future__ import annotations

import numpy as np

from waam_twin.measure.steady_section import steady_cross_section

NX, NY, NZ = 12, 9, 10
DX_MM = 0.4
NZ_SOLID = 4
# Three slices, five columns, three cells above the plate.
I0, I1 = 4, 6
J0, J1 = 2, 7


class _Field:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def to_numpy(self) -> np.ndarray:
        return self.values


class _Grid:
    dx = DX_MM * 1e-3
    nx, ny, nz = NX, NY, NZ
    FLAG_FLUID = 0
    FLAG_SOLID = 1
    FLAG_GAS = 2

    def __init__(self) -> None:
        self.T_max = _Field(np.full((NX, NY, NZ), 300.0, dtype=np.float32))
        self.flags = _Field(np.full((NX, NY, NZ), self.FLAG_GAS, dtype=np.int32))
        self.phi = _Field(np.zeros((NX, NY, NZ), dtype=np.float32))


class _Mat:
    T_liquidus = 1793.0


class _Twin:
    def __init__(self) -> None:
        self.grid = _Grid()
        self.nz_solid = NZ_SOLID
        self.plate_k0 = 0
        self.mat = _Mat()
        self.plate_mat = None


def _lay_bead(twin: _Twin, *, height: int = 3, phi: float = 1.0, flag: int | None = None) -> None:
    g = twin.grid
    solid = g.FLAG_SOLID if flag is None else flag
    g.flags.values[I0:I1 + 1, J0:J1, NZ_SOLID:NZ_SOLID + height] = solid
    g.phi.values[I0:I1 + 1, J0:J1, NZ_SOLID:NZ_SOLID + height] = phi


def _measure(twin: _Twin) -> dict:
    return steady_cross_section(twin, I0 * DX_MM, I1 * DX_MM)


def flat_frozen() -> None:
    twin = _Twin()
    _lay_bead(twin)
    out = _measure(twin)
    cell = DX_MM * DX_MM
    if abs(out["H_frozen_mm"] - 1.2) > 1e-6:
        raise AssertionError(f"H_frozen {out['H_frozen_mm']}")
    if abs(out["area_mm2"] - 15 * cell) > 1e-6:
        raise AssertionError(f"area {out['area_mm2']}")
    if abs(out["W_bead_mm"] - 2.0) > 1e-6:
        raise AssertionError(f"W_bead {out['W_bead_mm']}")
    if not out["steady"]:
        raise AssertionError(f"expected steady, frozen_frac={out['frozen_frac']}")


def liquid_hump() -> None:
    twin = _Twin()
    _lay_bead(twin)
    g = twin.grid
    # One column in the middle slice is liquid and two cells taller.
    g.flags.values[5, 4, NZ_SOLID:NZ_SOLID + 5] = g.FLAG_FLUID
    g.phi.values[5, 4, NZ_SOLID:NZ_SOLID + 5] = 1.0
    out = _measure(twin)
    cell = DX_MM * DX_MM
    # The hump slice is 2.0 mm; the other two stay 1.2 mm.
    expected_h = (1.2 + 2.0 + 1.2) / 3.0
    if abs(out["H_mm"] - expected_h) > 1e-6:
        raise AssertionError(f"H_mm {out['H_mm']} != {expected_h} (hump slice is 2.0 mm)")
    if abs(out["H_frozen_mm"] - 1.2) > 1e-6:
        raise AssertionError(f"H_frozen moved to {out['H_frozen_mm']}")
    expected_area = (3 * 15 * cell + 2 * cell) / 3.0
    if abs(out["area_mm2"] - expected_area) > 1e-6:
        raise AssertionError(f"area {out['area_mm2']} != {expected_area}")
    if out["steady"]:
        raise AssertionError("liquid column should fail the steady gate")


def partial_fill() -> None:
    twin = _Twin()
    _lay_bead(twin)
    twin.grid.phi.values[I0:I1 + 1, J0:J1, NZ_SOLID + 2] = 0.5
    out = _measure(twin)
    if abs(out["H_frozen_mm"] - 1.0) > 1e-6:
        raise AssertionError(f"H_frozen {out['H_frozen_mm']}")


def run() -> None:
    flat_frozen()
    liquid_hump()
    partial_fill()
    print("[steady_section] flat frozen, liquid hump, partial fill")
    print("PASS")


if __name__ == "__main__":
    run()
