"""Index-ordered reductions (C-order flat index, block length 256).

Block sums run in parallel; each block walks its cells in ascending flat
index. A serialized pass then adds the block partials in block order. The
total matches ``float64(field.to_numpy().ravel(order="C")).sum()`` bit for
bit when the field values are exactly representable in the scratch dtype.
"""

from __future__ import annotations

from waam_twin.compiler import ti

REDUCE_BLOCK = 256


@ti.kernel
def zero_reduce_scratch(scratch: ti.template()):
    for i, j, k in scratch:
        scratch[i, j, k] = 0.0


@ti.kernel
def sum_blocks_ordered(
    scratch: ti.template(),
    partials: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    n_cells: ti.i32,
    n_blocks: ti.i32,
    block: ti.i32,
):
    """Each block writes partials[b] = sum of scratch in C-order over its range."""
    for b in range(n_blocks):
        s = ti.cast(0.0, ti.f64)
        base = b * block
        for t in range(block):
            idx = base + t
            if idx < n_cells:
                i = idx // (ny * nz)
                rem = idx - i * (ny * nz)
                j = rem // nz
                k = rem - j * nz
                s += ti.cast(scratch[i, j, k], ti.f64)
        partials[b] = s


@ti.kernel
def sum_partials_ordered(
    partials: ti.template(),
    out: ti.template(),
    n_blocks: ti.i32,
):
    """Serialized sum of block partials into a 0-d f64 field (out starts at 0)."""
    out[None] = 0.0
    ti.loop_config(serialize=True)
    for b in range(n_blocks):
        out[None] += partials[b]


@ti.kernel
def copy_total_to_f32(src: ti.template(), dst: ti.template()):
    dst[None] = ti.cast(src[None], ti.f32)


@ti.kernel
def add_total_to_acc(total: ti.template(), acc: ti.template(), slot: ti.i32):
    acc[slot] += total[None]


def n_reduce_blocks(n_cells: int, block: int = REDUCE_BLOCK) -> int:
    return max(1, (int(n_cells) + int(block) - 1) // int(block))


def ordered_sum_field(g, field) -> float:
    """Sum an f32 volume in C-order into ``g.reduce_total``; return it."""
    n_cells = int(g.nx) * int(g.ny) * int(g.nz)
    n_blocks = n_reduce_blocks(n_cells)
    sum_blocks_ordered(
        field, g.reduce_partials,
        g.nx, g.ny, g.nz, n_cells, n_blocks, REDUCE_BLOCK,
    )
    sum_partials_ordered(g.reduce_partials, g.reduce_total, n_blocks)
    return float(g.reduce_total[None])


def ordered_sum_scratch(g) -> float:
    """Sum ``g.reduce_scratch`` in C-order into ``g.reduce_total``; return it."""
    return ordered_sum_field(g, g.reduce_scratch)
