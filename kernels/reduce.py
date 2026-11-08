"""Fixed-order (deterministic) f64 reductions of f32 volume fields.

``ordered_sum_field`` sums over i per (j, k) column (coalesced), then over k
per j, then serially over j. The order never changes, so repeated runs give
identical bits; for f32 inputs the f64 total equals NumPy's whenever the sum
is exact in f64 (the usual case for field sizes here). The older
block-of-256 kernels (``sum_blocks_ordered``) are kept for API use.
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


@ti.kernel
def sum_plane_over_i(
    scratch: ti.template(), plane: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32,
):
    """plane[j, k] = Σ_i scratch[i, j, k] in ascending i (f64).

    Adjacent threads take adjacent k, so each step of the i loop reads one
    contiguous run of memory (coalesced), unlike a thread per 256-cell block.
    """
    for j, k in ti.ndrange(ny, nz):
        s = ti.cast(0.0, ti.f64)
        for i in range(nx):
            s += ti.cast(scratch[i, j, k], ti.f64)
        plane[j, k] = s


@ti.kernel
def sum_lines_over_k(plane: ti.template(), line: ti.template(), ny: ti.i32, nz: ti.i32):
    for j in range(ny):
        s = ti.cast(0.0, ti.f64)
        for k in range(nz):
            s += plane[j, k]
        line[j] = s


@ti.kernel
def sum_line_serial(line: ti.template(), out: ti.template(), ny: ti.i32):
    out[None] = 0.0
    ti.loop_config(serialize=True)
    for j in range(ny):
        out[None] += line[j]


@ti.kernel
def copy_total_to_f64(src: ti.template(), dst: ti.template()):
    dst[None] = src[None]


def n_reduce_blocks(n_cells: int, block: int = REDUCE_BLOCK) -> int:
    return max(1, (int(n_cells) + int(block) - 1) // int(block))


def ordered_sum_field(g, field, *, to_host: bool = True):
    """Sum an f32 volume into ``g.reduce_total`` in a fixed order (f64).

    Order: over i for each (j, k), then over k for each j, then over j —
    the same every run, so results are deterministic. With ``to_host`` the
    total is also returned as a Python float, which waits for the GPU;
    the stepper passes ``to_host=False`` and keeps the value on the device.
    """
    sum_plane_over_i(field, g.reduce_plane, g.nx, g.ny, g.nz)
    sum_lines_over_k(g.reduce_plane, g.reduce_line, g.ny, g.nz)
    sum_line_serial(g.reduce_line, g.reduce_total, g.ny)
    if to_host:
        return float(g.reduce_total[None])
    return None


def ordered_sum_scratch(g, *, to_host: bool = True):
    """Sum ``g.reduce_scratch`` into ``g.reduce_total`` (see ordered_sum_field)."""
    return ordered_sum_field(g, g.reduce_scratch, to_host=to_host)
