"""Host side of the sparse grid: which blocks exist, and when they change.

Rules (see kernels/sparse.py):
  - reset: blocks over the plate plus ``sparse_halo_blocks`` (all blocks for
    an all-fluid test domain or a plate that fills the domain);
  - every ``sparse_refresh_every`` steps: blocks holding metal plus the halo
    are kept, new ones get the dense-grid gas state, far all-gas ones go;
  - before a droplet is placed: the deposition search box is activated.

Between refreshes metal moves < 2 cells (Mach cap 0.08 cell/step × 16 steps),
far less than the 8-cell halo, so nothing reaches a missing block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kernels import sparse as ks

if TYPE_CHECKING:
    from ..twin import WAAMTwin


def check_supported(twin: "WAAMTwin") -> None:
    """Refuse modes whose gas cells carry state or that copy the whole grid."""
    g = twin.grid
    if not getattr(g, "sparse", False):
        return
    if twin.enable_vof and not getattr(twin, "use_free_surface_lbm", False):
        raise ValueError(
            "sparse_grid needs use_free_surface_lbm (legacy VOF keeps populations in gas cells)"
        )
    if getattr(twin, "enable_moving_window", False):
        raise ValueError("sparse_grid does not support the moving window yet")
    if getattr(twin, "periodic_x", False):
        raise ValueError("sparse_grid does not support periodic_x")


def _blocks(g, lo: int, hi: int, axis: int) -> tuple[int, int]:
    B = g.sparse_block
    nb = g.nb[axis]
    return max(0, lo // B), min(nb, -(-hi // B))


def activate_cell_box(twin: "WAAMTwin", i0: int, i1: int, j0: int, j1: int, k0: int, k1: int) -> None:
    """Make every block touching cells [i0,i1)×[j0,j1)×[k0,k1) exist (gas-initialised)."""
    g = twin.grid
    b0, b1 = _blocks(g, i0, i1, 0)
    c0, c1 = _blocks(g, j0, j1, 1)
    d0, d1 = _blocks(g, k0, k1, 2)
    if b1 > b0 and c1 > c0 and d1 > d0:
        ks.activate_block_box(g._ptr3, g._ptr4, b0, b1, c0, c1, d0, d1)
        init_fresh_gas(twin)


def init_fresh_gas(twin: "WAAMTwin") -> None:
    g = twin.grid
    T0, rho0, H_gas = twin.__dict__.get("_sparse_gas_state", (float(twin.T_amb), 1.0, 0.0))
    ks.init_fresh_gas(
        g.f_a, g.f_b, g.rho, g.ux, g.uy, g.uz, g.T, g.H, g.f_l, g.phi,
        g.flags, g.alloy_id, g.alloy_frac, g.T_max, g.T_prev,
        T0, rho0, H_gas, g.FLAG_GAS,
    )


def activate_for_reset(twin: "WAAMTwin", all_blocks: bool, plate_box: tuple[int, ...]) -> None:
    """Blocks that must exist before init_grid (which only visits active blocks)."""
    g = twin.grid
    if all_blocks:
        _check_pool(twin, g.nb[0] * g.nb[1] * g.nb[2])
        ks.activate_block_box(g._ptr3, g._ptr4, 0, g.nb[0], 0, g.nb[1], 0, g.nb[2])
        return
    i0, i1, j0, j1, k0, k1 = plate_box
    h = int(getattr(twin, "sparse_halo_blocks", 1)) * g.sparse_block
    b0, b1 = _blocks(g, i0 - h, i1 + h, 0)
    c0, c1 = _blocks(g, j0 - h, j1 + h, 1)
    d0, d1 = _blocks(g, k0 - h, k1 + h, 2)
    _check_pool(twin, (b1 - b0) * (c1 - c0) * (d1 - d0))
    ks.activate_block_box(g._ptr3, g._ptr4, b0, b1, c0, c1, d0, d1)


def refresh(twin: "WAAMTwin", allow_deactivate: bool = True) -> None:
    """Keep metal blocks + halo, create new ones, drop far all-gas ones."""
    g = twin.grid
    g.blk_need.fill(0)
    ks.mark_metal_blocks(g.flags, g.blk_need, g.FLAG_GAS, g.sparse_block)
    ks.dilate_blocks(
        g.blk_need, g.blk_keep, int(getattr(twin, "sparse_halo_blocks", 1)),
        g.nb[0], g.nb[1], g.nb[2],
    )
    _check_pool(twin, int(ks.count_kept_blocks(g.blk_keep)))
    ks.apply_block_mask(g._ptr3, g._ptr4, g.blk_keep, 1 if allow_deactivate else 0)
    init_fresh_gas(twin)


def maybe_refresh(twin: "WAAMTwin") -> None:
    g = twin.grid
    if not getattr(g, "sparse", False):
        return
    every = max(1, int(getattr(twin, "sparse_refresh_every", 16) or 1))
    if int(getattr(twin, "_step_n", 0)) % every == 0:
        refresh(twin)


def active_fraction(twin: "WAAMTwin") -> float:
    g = twin.grid
    if not getattr(g, "sparse", False):
        return 1.0
    n = int(ks.count_active_blocks(g._ptr3, g.blk_keep))
    return n / float(g.nb[0] * g.nb[1] * g.nb[2])


# Allocator overhead on top of field bytes: the 1 GB pool ran out between 0.83
# and 0.95 of its size by field bytes alone (RTX 3050, Nov 2026).
_POOL_MARGIN = 1.25


def pool_bytes() -> float | None:
    """Sparse allocation pool on the GPU, or None on CPU (no fixed pool)."""
    from quadrants.lang import impl
    from ..compiler import ti

    cfg = impl.current_cfg()
    if cfg.arch in (ti.cpu, ti.x64, ti.arm64):
        return None
    if cfg.device_memory_fraction and cfg.device_memory_fraction > 0:
        return None  # fraction of the device: let the driver decide
    return float(cfg.device_memory_GB) * 1024 ** 3


def _check_pool(twin: "WAAMTwin", n_blocks: int) -> None:
    """Stop with a clear message before the GPU pool fills (it would assert)."""
    pool = pool_bytes()
    if pool is None:
        return
    g = twin.grid
    need = n_blocks * float(g.sparse_bytes_per_block) * _POOL_MARGIN
    twin.__dict__["_sparse_pool_use"] = need / pool
    if need > 0.95 * pool:
        gb = need / 1024 ** 3
        raise RuntimeError(
            f"sparse grid needs ≈{gb:.2f} GB for {n_blocks} active blocks but the GPU "
            f"pool is {pool / 1024 ** 3:.2f} GB. Set WAAM_DEVICE_MEMORY_GB to about "
            f"{gb * 1.3:.1f} (or more for a growing wall) before starting."
        )
    if need > 0.8 * pool and not twin.__dict__.get("_sparse_pool_warned"):
        twin.__dict__["_sparse_pool_warned"] = True
        from .. import logging_util as log
        log.warning(
            f"[sparse_grid] active blocks use {need / pool * 100:.0f}% of the "
            f"{pool / 1024 ** 3:.2f} GB GPU pool; raise WAAM_DEVICE_MEMORY_GB for growing walls"
        )
