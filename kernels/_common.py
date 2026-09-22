"""Shared Quadrants helpers for waam_twin.kernels."""

from waam_twin.compiler import ti

from ..gpu_tables import MAX_KNOTS
from ..lattice import EX, EY, EZ, W, OPP

# Birth-alloy tags. alloy_id stays 0/1 for diagnostics; alloy_frac in [0, 1]
# is the composition used for properties (0=wire, 1=plate). Mixing is opt-in.
ALLOY_WIRE = 0
ALLOY_PLATE = 1


@ti.func
def _cell_rho(alloy_frac: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32,
              rho_wire: ti.f32, rho_plate: ti.f32) -> ti.f32:
    f = alloy_frac[i, j, k]
    return rho_wire * (1.0 - f) + rho_plate * f


def bind_velocity_set(grid):
    """Kept for API compatibility — velocity set is now compile-time constant."""
    return None

@ti.func
def lookup_table_1d(
    T: ti.f32,
    t_arr: ti.template(),
    v_arr: ti.template(),
    n: ti.i32,
    fallback: ti.f32,
) -> ti.f32:
    result = fallback
    if n > 0:
        if T <= t_arr[0]:
            result = v_arr[0]
        elif T >= t_arr[n - 1]:
            result = v_arr[n - 1]
        else:
            for idx in ti.static(range(MAX_KNOTS - 1)):
                if idx < n - 1:
                    t0 = t_arr[idx]
                    t1 = t_arr[idx + 1]
                    if T >= t0 and T <= t1:
                        w = (T - t0) / (t1 - t0 + 1e-8)
                        result = v_arr[idx] + w * (v_arr[idx + 1] - v_arr[idx])
    return result


@ti.func
def _alloy_pick(frac: ti.f32, wire_v: ti.f32, plate_v: ti.f32) -> ti.f32:
    """Lerp wire→plate by composition fraction (0=wire, 1=plate)."""
    return wire_v * (1.0 - frac) + plate_v * frac

