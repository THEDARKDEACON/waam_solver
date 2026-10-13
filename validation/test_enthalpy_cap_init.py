"""H6 regression: step-1 enthalpy cap must not wipe the plate."""

from __future__ import annotations

import os


def run() -> float:
    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin

    init_taichi(backend=os.environ.get("WAAM_BACKEND", "cpu"))
    twin = WAAMTwin.from_preset("minimal")
    twin.enable_enthalpy_cap = True
    twin.reset()
    g = twin.grid

    # Stored enthalpy after reset (should be ambient plate)
    import numpy as np
    H0 = float((g.H.to_numpy() * (g.flags.to_numpy() != g.FLAG_GAS)).sum() * g.dx ** 3)

    # One welding step at plate centre
    twin.step(0.5 * g.nx * g.dx, 0.5 * g.ny * g.dx, is_welding=True)
    cap = float(getattr(twin, "_enthalpy_cap_energy_J_cum", 0.0))
    print(f"[enthalpy_cap_init] H0={H0:.1f} J  cap_cum={cap:.3f} J")
    # Cap must not discard ~the whole plate energy on step 1
    if cap > 0.05 * max(H0, 1.0):
        raise AssertionError(
            f"H6: step-1 cap discarded {cap:.1f} J (~{100*cap/max(H0,1):.0f}% of plate) "
            f"— cp_rho_field was uninitialised"
        )
    return cap


if __name__ == "__main__":
    run()
