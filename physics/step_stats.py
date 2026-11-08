"""Device-side per-step counters (no host sync per step).

Energy removed by the enthalpy cap and evaporation, and force / Mach clamp
hits, are accumulated on the GPU in ``grid.step_stats`` (f64). The twin's
``_evap_energy_J_*``, ``_enthalpy_cap_energy_J_*``, ``_*_clamp_*`` attributes
read these slots on access, so telemetry is unchanged but the stepper no
longer waits for the GPU every time a counter is updated.
"""

from __future__ import annotations

from waam_twin.compiler import ti

CAP_STEP, CAP_CUM = 0, 1
EVAP_STEP, EVAP_CUM = 2, 3
FHITS_STEP, FHITS_CUM, FSTEPS = 4, 5, 6
MHITS_STEP, MHITS_CUM, MSTEPS = 7, 8, 9
N_SLOTS = 10

# twin attribute → (slot, type)
TWIN_ATTRS = {
    "_enthalpy_cap_energy_J_step": (CAP_STEP, float),
    "_enthalpy_cap_energy_J_cum": (CAP_CUM, float),
    "_evap_energy_J_step": (EVAP_STEP, float),
    "_evap_energy_J_cum": (EVAP_CUM, float),
    "_force_clamp_hits_step": (FHITS_STEP, int),
    "_force_clamp_hits_cum": (FHITS_CUM, int),
    "_force_clamp_steps": (FSTEPS, int),
    "_mach_clamp_hits_step": (MHITS_STEP, int),
    "_mach_clamp_hits_cum": (MHITS_CUM, int),
    "_mach_clamp_steps": (MSTEPS, int),
}


@ti.kernel
def reset_step_slots(stats: ti.template()):
    """Zero the per-step slots at the start of a step."""
    stats[CAP_STEP] = 0.0
    stats[EVAP_STEP] = 0.0
    stats[FHITS_STEP] = 0.0
    stats[MHITS_STEP] = 0.0


@ti.kernel
def add_energy(buf: ti.template(), stats: ti.template(), step_slot: ti.i32, cum_slot: ti.i32):
    e = ti.cast(buf[None], ti.f64)
    stats[step_slot] += e
    stats[cum_slot] += e


@ti.kernel
def add_hits(
    buf: ti.template(), stats: ti.template(),
    step_slot: ti.i32, cum_slot: ti.i32, steps_slot: ti.i32,
):
    n = ti.cast(buf[None], ti.f64)
    stats[step_slot] = n
    stats[cum_slot] += n
    if n > 0.0:
        stats[steps_slot] += 1.0


class DeviceStat:
    """Twin attribute backed by a ``grid.step_stats`` slot (host fallback before the grid exists)."""

    def __init__(self, slot: int, cast: type):
        self.slot = slot
        self.cast = cast
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    @staticmethod
    def _field(obj):
        g = obj.__dict__.get("grid")
        return getattr(g, "step_stats", None) if g is not None else None

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        f = self._field(obj)
        if f is None:
            return obj.__dict__.get("_ds_" + self.name, self.cast(0))
        v = float(f[self.slot])
        return int(round(v)) if self.cast is int else v

    def __set__(self, obj, value) -> None:
        f = self._field(obj)
        if f is None:
            obj.__dict__["_ds_" + self.name] = self.cast(value)
        else:
            f[self.slot] = float(value)


def device_energy_totals(twin) -> tuple[float, float]:
    """(enthalpy-cap J, evaporation J) accumulated on the device."""
    f = getattr(getattr(twin, "grid", None), "step_stats", None)
    if f is None:
        return 0.0, 0.0
    v = f.to_numpy()
    return float(v[CAP_CUM]), float(v[EVAP_CUM])
