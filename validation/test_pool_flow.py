"""Host-side checks for the pool-flow diagnostic. No GPU."""

from __future__ import annotations

import numpy as np

from waam_twin.measure.pool_flow import pool_flow_stats


class _Field:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def to_numpy(self) -> np.ndarray:
        return self.values


class _Grid:
    FLAG_FLUID = 0
    FLAG_SOLID = 1
    FLAG_GAS = 2
    FLAG_IFACE = 4
    dx = 4e-4
    dt = 4e-5
    tau = 0.6

    def __init__(self, ux: np.ndarray) -> None:
        shape = ux.shape
        self.flags = _Field(np.full(shape, self.FLAG_FLUID, dtype=np.int32))
        self.f_l = _Field(np.ones(shape, dtype=np.float32))
        self.ux = _Field(ux.astype(np.float32))
        self.uy = _Field(np.zeros(shape, dtype=np.float32))
        self.uz = _Field(np.zeros(shape, dtype=np.float32))
        self.flags.values[:, :, -1] = self.FLAG_GAS


class _Twin:
    def __init__(self, ux: np.ndarray) -> None:
        self.grid = _Grid(ux)


def uniform_shear() -> None:
    nx, ny, nz = 6, 5, 8
    shear = 0.02
    k = np.arange(nz, dtype=np.float64)
    ux = np.broadcast_to(shear * k, (nx, ny, nz)).copy()
    cs = 0.1
    out = pool_flow_stats(_Twin(ux), cs)
    nu_mol = (0.6 - 0.5) / 3.0
    expected = cs ** 2 * shear / nu_mol
    if abs(out["nu_ratio"] - expected) / expected > 0.02:
        raise AssertionError(f"nu_ratio {out['nu_ratio']} != {expected}")
    if out["u_pool_max_ms"] <= 0.0:
        raise AssertionError("shear field should have pool speed")
    if out["n_pool"] <= 0:
        raise AssertionError("pool should be non-empty")


def zero_field() -> None:
    ux = np.zeros((4, 4, 4), dtype=np.float64)
    out = pool_flow_stats(_Twin(ux), 0.17)
    if out["u_pool_max_ms"] != 0.0 or out["u_surf_p95_ms"] != 0.0:
        raise AssertionError(f"zero field has speed {out}")
    if out["nu_ratio"] != 0.0:
        raise AssertionError(f"zero field nu_ratio {out['nu_ratio']}")


def _surface_twin(fl_surface: float, c_darcy: float):
    ux = np.zeros((4, 4, 4), dtype=np.float64)
    twin = _Twin(ux)
    twin.grid.f_l.values[:, :, -2] = fl_surface
    twin.effective_C_darcy = lambda: c_darcy
    return twin


def liquid_surface() -> None:
    out = pool_flow_stats(_surface_twin(1.0, 160000.0), 0.0)
    if abs(out["surface_atten_median"] - 1.0) > 1e-6:
        raise AssertionError(f"liquid attenuation {out['surface_atten_median']}")
    if out["surface_frac_liquid"] < 0.99:
        raise AssertionError(f"liquid fraction {out['surface_frac_liquid']}")


def mush_surface() -> None:
    out = pool_flow_stats(_surface_twin(0.9, 160000.0), 0.0)
    if out["surface_atten_median"] <= 100.0:
        raise AssertionError(f"mush attenuation {out['surface_atten_median']}")


def run() -> None:
    uniform_shear()
    zero_field()
    liquid_surface()
    mush_surface()
    print("[pool_flow] uniform shear, zero field, liquid surface, mush surface")
    print("PASS")


if __name__ == "__main__":
    run()
