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

# ceil(log2(MAX_KNOTS - 1)) halvings bracket any interval of a full table.
_BISECT_STEPS = max(1, (MAX_KNOTS - 2).bit_length())


@ti.func
def _bracket(x: ti.f32, arr: ti.template(), n: ti.i32) -> ti.i32:
    """Largest idx in [0, n-2] with arr[idx] <= x, for ascending arr[0] < x < arr[n-1]."""
    lo = 0
    hi = n - 1
    for _ in ti.static(range(_BISECT_STEPS)):
        if hi - lo > 1:
            mid = (lo + hi) // 2
            if x >= arr[mid]:
                lo = mid
            else:
                hi = mid
    return lo


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
            idx = _bracket(T, t_arr, n)
            t0 = t_arr[idx]
            t1 = t_arr[idx + 1]
            w = (T - t0) / (t1 - t0 + 1e-8)
            result = v_arr[idx] + w * (v_arr[idx + 1] - v_arr[idx])
    return result


@ti.func
def _alloy_pick(frac: ti.f32, wire_v: ti.f32, plate_v: ti.f32) -> ti.f32:
    """Lerp wire→plate by composition fraction (0=wire, 1=plate)."""
    return wire_v * (1.0 - frac) + plate_v * frac


@ti.func
def _segment_temperature_offset(above: ti.f32, c0: ti.f32, c1: ti.f32, dt: ti.f32) -> ti.f32:
    """Invert one piecewise-linear cp interval. Constant cp returns above/c0."""
    u = above / (c0 + 1e-12)
    dc = c1 - c0
    if dt > 1e-8 and ti.abs(dc) > 1e-4:
        disc = c0 * c0 + 2.0 * dc / dt * above
        if disc > 0.0:
            u = 2.0 * above / (c0 + ti.sqrt(disc) + 1e-12)
    if u < 0.0:
        u = 0.0
    if dt > 1e-8 and u > dt:
        u = dt
    return u


@ti.func
def sensible_specific(
    T: ti.f32,
    t_arr: ti.template(),
    cp_arr: ti.template(),
    h_arr: ti.template(),
    n: ti.i32,
) -> ti.f32:
    """∫_0^T cp [J/kg] on the uploaded knots. h_arr[i] is that integral at t_arr[i]."""
    out = 0.0
    if n > 0 and T > 0.0:
        if T <= t_arr[0]:
            out = cp_arr[0] * T
        elif T >= t_arr[n - 1]:
            out = h_arr[n - 1] + cp_arr[n - 1] * (T - t_arr[n - 1])
        else:
            idx = _bracket(T, t_arr, n)
            t0 = t_arr[idx]
            t1 = t_arr[idx + 1]
            u = T - t0
            dt = t1 - t0
            c0 = cp_arr[idx]
            c1 = cp_arr[idx + 1]
            out = h_arr[idx] + c0 * u
            if dt > 1e-8:
                out = out + 0.5 * (c1 - c0) / dt * u * u
    return out


@ti.func
def invert_sensible(
    target: ti.f32,
    t_arr: ti.template(),
    cp_arr: ti.template(),
    h_arr: ti.template(),
    n: ti.i32,
) -> ti.f32:
    """Temperature whose sensible enthalpy is `target` [J/kg]."""
    T_out = 0.0
    if n > 0:
        if target <= h_arr[0]:
            T_out = target / (cp_arr[0] + 1e-12)
        elif target >= h_arr[n - 1]:
            T_out = t_arr[n - 1] + (target - h_arr[n - 1]) / (cp_arr[n - 1] + 1e-12)
        else:
            idx = _bracket(target, h_arr, n)
            dt = t_arr[idx + 1] - t_arr[idx]
            u = _segment_temperature_offset(
                target - h_arr[idx], cp_arr[idx], cp_arr[idx + 1], dt,
            )
            T_out = t_arr[idx] + u
    return T_out


@ti.func
def temperature_from_enthalpy(
    h_vol: ti.f32,
    rho: ti.f32,
    L_rho: ti.f32,
    T_sol: ti.f32,
    T_liq: ti.f32,
    t_arr: ti.template(),
    cp_arr: ti.template(),
    h_arr: ti.template(),
    n: ti.i32,
) -> ti.f32:
    """Invert rho·∫cp plus latent heat spread from T_sol to T_liq."""
    h_spec = h_vol / (rho + 1e-12)
    L_spec = L_rho / (rho + 1e-12)
    h_sol = sensible_specific(T_sol, t_arr, cp_arr, h_arr, n)
    h_liq = sensible_specific(T_liq, t_arr, cp_arr, h_arr, n)
    T_out = invert_sensible(h_spec, t_arr, cp_arr, h_arr, n)
    if h_spec > h_sol and h_spec < h_liq + L_spec and (T_liq - T_sol) > 1e-3:
        lo = T_sol
        hi = T_liq
        span = T_liq - T_sol
        for _it in range(24):
            mid = 0.5 * (lo + hi)
            acc = sensible_specific(mid, t_arr, cp_arr, h_arr, n) + L_spec * (mid - T_sol) / span
            if acc < h_spec:
                lo = mid
            else:
                hi = mid
        T_out = 0.5 * (lo + hi)
    elif h_spec >= h_liq + L_spec:
        T_out = invert_sensible(h_spec - L_spec, t_arr, cp_arr, h_arr, n)
    return T_out


@ti.func
def temperature_from_enthalpy_mix(
    h_vol: ti.f32,
    frac: ti.f32,
    rho_w: ti.f32,
    L_w: ti.f32,
    T_sol_w: ti.f32,
    T_liq_w: ti.f32,
    t_w: ti.template(),
    cp_w: ti.template(),
    h_w: ti.template(),
    n_w: ti.i32,
    rho_p: ti.f32,
    L_p: ti.f32,
    T_sol_p: ti.f32,
    T_liq_p: ti.f32,
    t_p: ti.template(),
    cp_p: ti.template(),
    h_p: ti.template(),
    n_p: ti.i32,
) -> ti.f32:
    """Pure cells use the exact curve. A mixture bisects the volume-weighted sum."""
    T_out = 0.0
    if frac <= 1e-4:
        T_out = temperature_from_enthalpy(
            h_vol, rho_w, L_w, T_sol_w, T_liq_w, t_w, cp_w, h_w, n_w,
        )
    elif frac >= 1.0 - 1e-4:
        T_out = temperature_from_enthalpy(
            h_vol, rho_p, L_p, T_sol_p, T_liq_p, t_p, cp_p, h_p, n_p,
        )
    else:
        T_sol = (1.0 - frac) * T_sol_w + frac * T_sol_p
        T_liq = (1.0 - frac) * T_liq_w + frac * T_liq_p
        lat = (1.0 - frac) * L_w + frac * L_p
        lo = 1.0
        hi = 6000.0
        for _it in range(20):
            mid = 0.5 * (lo + hi)
            sens = (1.0 - frac) * rho_w * sensible_specific(mid, t_w, cp_w, h_w, n_w)
            sens = sens + frac * rho_p * sensible_specific(mid, t_p, cp_p, h_p, n_p)
            acc = sens
            if mid >= T_liq:
                acc = sens + lat
            elif mid > T_sol:
                acc = sens + lat * (mid - T_sol) / (T_liq - T_sol + 1e-9)
            if acc < h_vol:
                lo = mid
            else:
                hi = mid
        T_out = 0.5 * (lo + hi)
    return T_out
