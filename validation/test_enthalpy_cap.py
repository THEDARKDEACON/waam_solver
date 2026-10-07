"""
test_enthalpy_cap.py — Peak temperature stays below vaporization cap;
discarded ceiling energy is ledgered.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.runtime import init_taichi
from waam_twin.solvers.coupled_step import _clamp_enthalpy_ceiling


def _assert_ceiling_ledger(twin: WAAMTwin) -> float:
    """Force one metal cell above the vapor-cap enthalpy and check the ledger."""
    g = twin.grid
    found = None
    for i in range(g.nx):
        for j in range(g.ny):
            for k in range(g.nz):
                if int(g.flags[i, j, k]) != g.FLAG_GAS:
                    found = (i, j, k)
                    break
            if found:
                break
        if found:
            break
    if found is None:
        raise AssertionError("no metal cell for enthalpy-cap ledger probe")
    i, j, k = found
    twin._enthalpy_cap_energy_J_step = 0.0
    before_cum = float(getattr(twin, "_enthalpy_cap_energy_J_cum", 0.0))
    # Overshoot liquidus enthalpy by a large sensible increment.
    g.H[i, j, k] = float(twin.H_liq + float(twin.cp_rho) * 2500.0)
    _clamp_enthalpy_ceiling(twin, g)
    e = float(twin._enthalpy_cap_energy_J_step)
    if e <= 0.0:
        raise AssertionError(
            f"enthalpy ceiling discarded energy was not ledgered (cell={found}, e={e})"
        )
    if float(twin._enthalpy_cap_energy_J_cum) <= before_cum:
        raise AssertionError("enthalpy_cap_energy_J_cum did not increase")
    return e


def run(max_peak_K: float = 3250.0, n_steps: int = 400) -> float:
    init_taichi(backend="cpu")
    twin = WAAMTwin(
        nx=24, ny=16, nz=16, dx=3e-4,
        enable_vof=True,
        enable_enthalpy_cap=True,
        T_vapor_cap_K=3200.0,
        arc_surface_weighting=True,
        heat_source="goldak",
        max_tracers=10,
    )
    twin.reset()
    e_probe = _assert_ceiling_ledger(twin)
    g = twin.grid
    cy = (g.ny // 2) * g.dx
    for step in range(n_steps):
        x = 0.003 + step * g.dt * 0.008
        twin.step(x, cy, is_welding=True)
    peak = float(g.T.to_numpy().max())
    telem = twin.get_telemetry()
    e_cap = float(telem.get("enthalpy_cap_energy_J_cum", 0.0))
    print(
        f"[enthalpy_cap] T_peak={peak:.0f}K  cap={twin.T_vapor_cap_K:.0f}K  "
        f"ledger_probe={e_probe:.4f}J  cap_energy_cum={e_cap:.4f}J"
    )
    if peak > max_peak_K:
        raise AssertionError(f"Peak T {peak:.0f}K exceeds cap band {max_peak_K:.0f}K")
    if e_cap <= 0.0:
        raise AssertionError("telemetry enthalpy_cap_energy_J_cum lost after steps")
    return peak


if __name__ == "__main__":
    run()
    print("PASS")
