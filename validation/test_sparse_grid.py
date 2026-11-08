"""
test_sparse_grid.py — sparse block grid (simulation.sparse_grid).

The sparse grid is GPU-only: Quadrants 1.3.3's CPU struct-for over pointer
blocks skips cells (96 of 24576 here), so on a CPU backend sparse_grid falls
back to the dense grid. This test

  CPU: checks the fallback, and reports whether the Quadrants CPU bug is
       still present (so the restriction can be lifted when it is fixed);
  GPU (CUDA, when available):
    1. every cell of an active block is visited exactly once (3-D fields and
       the f[q,i,j,k] layout);
    2. missing blocks read as gas; reset activates only the plate + halo;
    3. the active set follows metal: metal placed high in the domain gets
       its block and halo on refresh (gas-initialised), released when gone;
    4. the GPU-pool guard stops with a clear message instead of an assert;
    5. dense vs sparse on a coupled FSLBM job: identical state after reset,
       and after 30 steps identical cell types with metal enthalpy equal
       within f32 summation-order noise (GPU atomics, block visit order).
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from waam_twin.compiler import ti
from waam_twin.paths import PROJECT_ROOT


@ti.kernel
def _mark_visits3(f: ti.template()):
    for i, j, k in f:
        f[i, j, k] = f[i, j, k] + 1.0


@ti.kernel
def _count_entries4(f: ti.template()) -> ti.i32:
    n = 0
    for q, i, j, k in f:
        n += 1
    return n


@ti.kernel
def _activate_box(p3: ti.template()):
    for bi, bj, bk in ti.ndrange(6, 4, 2):
        ti.activate(p3, [bi, bj, bk])


def _cuda_available() -> bool:
    import subprocess
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _small_grid():
    from waam_twin.grid import WAAMGrid
    from waam_twin.materials import load_material
    return WAAMGrid(48, 32, 32, 4e-4, load_material("ER70S-6"), max_tracers=4, sparse=True)


def _coverage(g) -> tuple[int, int]:
    """(unvisited, multiply visited) cells of active blocks for a 3-D field."""
    from waam_twin.kernels import sparse as ks
    ks.activate_block_box(g._ptr3, g._ptr4, 0, g.nb[0], 0, g.nb[1], 0, 2)
    g.T.fill(0.0)
    _mark_visits3(g.T)
    a = g.T.to_numpy()[:, :, : 2 * g.sparse_block]
    return int((a == 0).sum()), int((a > 1).sum())


def _cpu_part() -> None:
    from waam_twin.runtime import init_taichi, reset_taichi
    reset_taichi()
    init_taichi(backend="cpu")
    g = _small_grid()
    if g.sparse:
        raise AssertionError("sparse_grid must fall back to dense on a CPU backend")
    print("[sparse_grid] CPU backend: sparse_grid falls back to the dense grid")
    # Is the Quadrants CPU sparse loop still broken? Probe with a raw tree.
    fb = ti.FieldsBuilder()
    p3 = fb.pointer(ti.ijk, (6, 4, 4))
    f = ti.field(ti.f32)
    p3.dense(ti.ijk, 8).place(f)
    for _ in range(40):  # siblings, as in the real grid
        p3.dense(ti.ijk, 8).place(ti.field(ti.f32))
    fb.finalize()

    _activate_box(p3)
    _mark_visits3(f)
    a = f.to_numpy()[:, :, :16]
    miss = int((a == 0).sum())
    print(f"[sparse_grid] Quadrants CPU sparse struct-for: {miss} of {a.size} cells skipped"
          + ("  (bug still present: keep sparse GPU-only)" if miss else "  (fixed upstream: CPU could be enabled)"))


def _job(sparse: bool) -> str:
    from waam_twin.job import load_job_config
    job = load_job_config(PROJECT_ROOT / "jobs/examples/bead_calibrate.yaml")
    job["simulation"].update(
        strict_mode=False, domain_mm=[17.0, 10.6, 12.0], backend="cuda", sparse_grid=sparse,
    )
    job["plate"]["thickness_mm"] = 3.0
    job["plate"]["size_mm"] = [14, 8]
    job["torch_path"] = [{"x_mm": 4.0, "y_mm": 5.2, "z_mm": 0.0}, {"x_mm": 14.0, "y_mm": 5.2, "z_mm": 0.0}]
    job.pop("torch_path_csv", None)
    tmp = PROJECT_ROOT / ".waam_sweep_tmp"
    tmp.mkdir(exist_ok=True)
    p = tmp / f"test_sparse_{int(sparse)}.yaml"
    p.write_text(yaml.safe_dump(job), encoding="utf-8")
    return str(p)


def _activation(twin) -> None:
    from waam_twin.physics import sparse_grid as sg
    g = twin.grid
    f0 = sg.active_fraction(twin)
    if not (0.0 < f0 < 1.0):
        raise AssertionError(f"reset should activate plate + halo only, got {f0:.2f}")
    probe = (g.nx // 2, g.ny // 2, g.nz - 2)
    if int(g.flags[probe]) != g.FLAG_GAS:
        raise AssertionError("a missing block must read as gas")
    g.flags[probe] = g.FLAG_SOLID
    sg.refresh(twin)
    f1 = sg.active_fraction(twin)
    gas_cell = (probe[0] + 3, probe[1], probe[2] - 3)
    rho, T = float(g.rho[gas_cell]), float(g.T[gas_cell])
    T0 = twin._sparse_gas_state[0]
    print(f"[sparse_grid] active {f0 * 100:.1f}% → metal at top → {f1 * 100:.1f}%  "
          f"fresh gas rho={rho:.3f} T={T:.1f}")
    if f1 <= f0:
        raise AssertionError("refresh did not create blocks around new metal")
    if abs(rho - 1.0) > 1e-6 or abs(T - T0) > 1e-3:
        raise AssertionError(f"fresh gas not initialised: rho={rho} T={T}")
    g.flags[probe] = g.FLAG_GAS
    sg.refresh(twin)
    f2 = sg.active_fraction(twin)
    if abs(f2 - f0) > 1e-9:
        raise AssertionError(f"blocks around removed metal not released: {f2:.3f} vs {f0:.3f}")


def _pool_guard(twin) -> None:
    from waam_twin.physics import sparse_grid as sg
    orig = sg.pool_bytes
    sg.pool_bytes = lambda: 1.0e6
    try:
        sg.refresh(twin)
    except RuntimeError as exc:
        if "WAAM_DEVICE_MEMORY_GB" not in str(exc):
            raise AssertionError(f"pool guard message lacks the fix: {exc}") from exc
        print("[sparse_grid] pool guard stops cleanly")
    else:
        raise AssertionError("pool guard did not trigger on a 1 MB pool")
    finally:
        sg.pool_bytes = orig


def _state(twin) -> dict:
    g = twin.grid
    nx, ny, nz = g.nx_req, g.ny_req, g.nz_req
    fl = g.flags.to_numpy()[:nx, :ny, :nz]
    cat = np.select(
        [fl == g.FLAG_GAS, fl == g.FLAG_FLUID, fl == g.FLAG_SOLID, fl == g.FLAG_IFACE],
        [0, 1, 2, 3], -1,
    )
    return {"cat": cat, "H": g.H.to_numpy()[:nx, :ny, :nz].astype(np.float64)}


def _run(sparse: bool, steps: int) -> tuple[dict, dict]:
    from waam_twin import WAAMTwin
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver
    path = _job(sparse)
    twin = WAAMTwin.from_job(path)
    twin.strict_mode = False
    twin.surface_query_every = 1
    twin.reset()
    if sparse and not twin.grid.sparse:
        raise AssertionError("sparse grid not active on the GPU backend")
    s0 = _state(twin)
    drv = TorchPathDriver(parse_torch_path(load_job_config(path)), twin.travel_speed_m_s)
    for _s, x, y, _z in drv.positions_for_steps(steps, twin.grid.dt):
        twin.step(x, y, is_welding=True)
    s1 = _state(twin)
    if sparse:
        _activation(twin)
        _pool_guard(twin)
    return s0, s1


def _compare(a: dict, b: dict) -> tuple[bool, float]:
    same = bool(np.array_equal(a["cat"], b["cat"]))
    metal = a["cat"] != 0
    rel = float(np.max(np.abs(a["H"] - b["H"])[metal] / np.maximum(np.abs(a["H"][metal]), 1.0)))
    return same, rel


def _gpu_part(steps: int) -> float | None:
    from quadrants.lang import impl
    from waam_twin.runtime import init_taichi, reset_taichi
    reset_taichi()
    init_taichi(backend="cuda")
    if impl.current_cfg().arch != ti.cuda:  # e.g. WAAM_BACKEND=cpu pins the suite
        print("[sparse_grid] backend pinned to non-CUDA: GPU checks skipped")
        return None
    g = _small_grid()
    miss, multi = _coverage(g)
    n4 = int(_count_entries4(g.f_a)) // (g.nb[0] * g.nb[1] * 2)
    print(f"[sparse_grid] GPU coverage: unvisited={miss} multi={multi}; populations {n4} per block")
    if miss or multi or n4 != 19 * g.sparse_block ** 3:
        raise AssertionError("GPU sparse struct-for does not visit every cell exactly once")
    d0, d1 = _run(False, steps)
    s0, s1 = _run(True, steps)
    same0, rel0 = _compare(d0, s0)
    same1, rel1 = _compare(d1, s1)
    print(f"[sparse_grid] dense vs sparse: after reset types={same0} relΔH={rel0:.1e};  "
          f"after {steps} steps types={same1} relΔH={rel1:.1e}")
    if not same0 or rel0 != 0.0:
        raise AssertionError("sparse reset differs from dense")
    if not same1 or rel1 > 1e-4:
        raise AssertionError(f"sparse run differs from dense beyond f32 order noise ({rel1:.1e})")
    return rel1


def run(steps: int = 30) -> float:
    rel = 0.0
    try:
        _cpu_part()
        if _cuda_available():
            rel = _gpu_part(steps) or 0.0
        else:
            print("[sparse_grid] no CUDA device: GPU checks skipped")
    finally:
        from waam_twin.runtime import init_taichi, reset_taichi
        reset_taichi()
        init_taichi(backend="cpu")
    print("[sparse_grid] PASS")
    return rel


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
