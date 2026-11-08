"""Block activation for the sparse grid (``WAAMGrid(sparse=True)``).

Volume fields live in 8³-cell blocks under two pointer SNodes with identical
block coordinates (``_ptr3`` for i,j,k fields, ``_ptr4`` for f[q,i,j,k]).
Only active blocks are stored and visited by ``for i, j, k in field`` loops;
a missing block reads as 0, and FLAG_GAS is 0 on a sparse grid, so it reads
as gas.

The active set is every block that holds metal (any non-gas cell) plus a
halo of blocks around it, refreshed every few steps, plus explicit boxes
(deposition) activated before they are written. Newly created blocks are
zero; ``init_fresh_gas`` gives their gas cells the state a dense grid's gas
cells have after ``reset`` (they are marked by rho == 0, which an initialised
cell never has).
"""

from __future__ import annotations

from waam_twin.compiler import ti

from ..lattice import W


@ti.kernel
def mark_metal_blocks(flags: ti.template(), need: ti.template(), FLAG_GAS: ti.i32, B: ti.i32):
    """need[b] = 1 for every block holding a non-gas cell (need must be zeroed)."""
    for i, j, k in flags:
        if flags[i, j, k] != FLAG_GAS:
            need[i // B, j // B, k // B] = ti.u8(1)


@ti.kernel
def dilate_blocks(
    need: ti.template(), keep: ti.template(), halo: ti.i32,
    nbx: ti.i32, nby: ti.i32, nbz: ti.i32,
):
    """keep[b] = 1 when any block within ±halo (Chebyshev) is needed."""
    for bi, bj, bk in keep:
        v = ti.u8(0)
        for di in range(-halo, halo + 1):
            for dj in range(-halo, halo + 1):
                for dk in range(-halo, halo + 1):
                    ni = bi + di
                    nj = bj + dj
                    nk = bk + dk
                    if 0 <= ni < nbx and 0 <= nj < nby and 0 <= nk < nbz:
                        if need[ni, nj, nk] != 0:
                            v = ti.u8(1)
        keep[bi, bj, bk] = v


@ti.kernel
def apply_block_mask(ptr3: ti.template(), ptr4: ti.template(), keep: ti.template(), allow_deactivate: ti.i32):
    """Activate kept blocks; deactivate the rest (when allowed)."""
    for bi, bj, bk in keep:
        if keep[bi, bj, bk] != 0:
            if not ti.is_active(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)]):
                ti.activate(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])
                ti.activate(ptr4, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])
        elif allow_deactivate != 0:
            if ti.is_active(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)]):
                ti.deactivate(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])
                ti.deactivate(ptr4, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])


@ti.kernel
def activate_block_box(
    ptr3: ti.template(), ptr4: ti.template(),
    b0: ti.i32, b1: ti.i32, c0: ti.i32, c1: ti.i32, d0: ti.i32, d1: ti.i32,
):
    """Activate blocks [b0,b1) × [c0,c1) × [d0,d1)."""
    for bi, bj, bk in ti.ndrange((b0, b1), (c0, c1), (d0, d1)):
        if not ti.is_active(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)]):
            ti.activate(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])
            ti.activate(ptr4, [ti.i32(bi), ti.i32(bj), ti.i32(bk)])


@ti.kernel
def count_active_blocks(ptr3: ti.template(), keep: ti.template()) -> ti.i32:
    n = 0
    for bi, bj, bk in keep:
        if ti.is_active(ptr3, [ti.i32(bi), ti.i32(bj), ti.i32(bk)]):
            n += 1
    return n


@ti.kernel
def init_fresh_gas(
    f_a: ti.template(), f_b: ti.template(),
    rho: ti.template(), ux: ti.template(), uy: ti.template(), uz: ti.template(),
    T: ti.template(), H: ti.template(), f_l: ti.template(), phi: ti.template(),
    flags: ti.template(), alloy_id: ti.template(), alloy_frac: ti.template(),
    T_max: ti.template(), T_prev: ti.template(),
    T_ambient: ti.f32, rho0: ti.f32, H_gas: ti.f32, FLAG_GAS: ti.i32,
):
    """Gas state of ``init_grid`` / ``init_aux_fields`` for never-initialised cells."""
    for i, j, k in rho:
        if flags[i, j, k] == FLAG_GAS and rho[i, j, k] == 0.0:
            rho[i, j, k] = rho0
            ux[i, j, k] = 0.0
            uy[i, j, k] = 0.0
            uz[i, j, k] = 0.0
            T[i, j, k] = T_ambient
            H[i, j, k] = H_gas
            f_l[i, j, k] = 0.0
            phi[i, j, k] = 0.0
            alloy_id[i, j, k] = 0
            alloy_frac[i, j, k] = 0.0
            T_max[i, j, k] = T_ambient
            T_prev[i, j, k] = T_ambient
            for q in ti.static(range(19)):
                f_a[q, i, j, k] = W[q] * rho0
                f_b[q, i, j, k] = W[q] * rho0


@ti.kernel
def copy_active(dst: ti.template(), src: ti.template()):
    """dst = src over existing cells (``Field.copy_from`` writes every index,
    which would create every block of a sparse grid)."""
    for I in ti.grouped(src):
        dst[I] = src[I]


@ti.kernel
def count_kept_blocks(keep: ti.template()) -> ti.i32:
    n = 0
    for bi, bj, bk in keep:
        if keep[bi, bj, bk] != 0:
            n += 1
    return n
