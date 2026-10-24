"""Ordered reduction and bit-identical Goldak deposit."""
from __future__ import annotations

import sys

import numpy as np

from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin, kernels
from waam_twin.kernels.reduce import ordered_sum_field, REDUCE_BLOCK
from waam_twin.physics.arc import Goldak3D


def test_ordered_sum_matches_numpy() -> None:
    twin = WAAMTwin(nx=17, ny=13, nz=11, dx=3e-4, max_tracers=4)
    g = twin.grid
    rng = np.random.default_rng(0)
    data = rng.standard_normal((g.nx, g.ny, g.nz), dtype=np.float32)
    data[::3, ::2, ::4] = 0.0
    g.reduce_scratch.from_numpy(data)
    got = ordered_sum_field(g, g.reduce_scratch)
    expect = float(data.astype(np.float64).ravel(order="C").sum())
    if got != expect:
        raise AssertionError(f"ordered_sum {got!r} != numpy {expect!r}  d={got - expect!r}")
    print(f"[ordered_reduce] sum matches numpy  got={got!r}  block={REDUCE_BLOCK}")


def test_goldak_bit_identical() -> None:
    twin = WAAMTwin(nx=24, ny=16, nz=16, dx=4e-4, max_tracers=4)
    twin.reset(T_ambient=300.0, test_fluid_domain=True)
    g = twin.grid
    phi = np.ones((g.nx, g.ny, g.nz), np.float32)
    flags = np.full((g.nx, g.ny, g.nz), g.FLAG_FLUID, np.int32)
    flags[:, :, -2:] = g.FLAG_GAS
    phi[:, :, -2:] = 0.0
    g.phi.from_numpy(phi)
    g.flags.from_numpy(flags)
    g.f_l.from_numpy((flags != g.FLAG_GAS).astype(np.float32))
    H0 = np.full((g.nx, g.ny, g.nz), 1.0e9, np.float32)
    g.H.from_numpy(H0.copy())
    src = Goldak3D(
        ff=0.6, fr=1.4,
        a_front=4.0, a_rear=6.0, b=5.0, c=4.0,
        normalize_energy=True,
    )
    twin.arc_source = src
    twin.Q_w = 5000.0
    twin.eta = 0.8
    arc = (g.nx * 0.5, g.ny * 0.5, float(g.nz) - 3.0)
    src.inject(twin, g, *arc)
    H1 = g.H.to_numpy().copy()
    g.H.from_numpy(H0.copy())
    src.inject(twin, g, *arc)
    H2 = g.H.to_numpy().copy()
    if not np.array_equal(H1, H2):
        raise AssertionError("Goldak H bits differ between identical injects")
    if np.array_equal(H1, H0):
        raise AssertionError("Goldak did not deposit")
    print("[ordered_reduce] goldak bit-identical")


def run() -> None:
    init_taichi()
    test_ordered_sum_matches_numpy()
    test_goldak_bit_identical()


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
