"""Field health checks.

The solver is compiled with fast-math, which assumes NaN never occurs, so a
``x != x`` test compiles to false. Max/min reductions also skip NaN (every
comparison with it is false): a blown-up field still reports a finite peak
temperature. These tests read the float's exponent bits instead.
"""

from __future__ import annotations

from waam_twin.compiler import ti

_EXP = 0x7F800000  # float32 exponent all ones = NaN or ±inf


@ti.kernel
def count_nonfinite(T: ti.template(), H: ti.template()) -> ti.i32:
    """Cells whose T or H is NaN or ±inf."""
    n = 0
    for i, j, k in T:
        bt = ti.bit_cast(T[i, j, k], ti.u32) & ti.u32(_EXP)
        bh = ti.bit_cast(H[i, j, k], ti.u32) & ti.u32(_EXP)
        if bt == ti.u32(_EXP) or bh == ti.u32(_EXP):
            n += 1
    return n
