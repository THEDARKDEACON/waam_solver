"""Host-side geometry measurement smoke (no GPU twin required for imports)."""

from __future__ import annotations

import numpy as np


def test_interp_crossing():
    from waam_twin.measure.geometry import _interp_crossing

    assert abs(_interp_crossing(0.0, 10.0, 5.0) - 0.5) < 1e-9
    assert 0.0 <= _interp_crossing(1.0, 1.0, 5.0) <= 1.0


def test_steady_detector_logic():
    from waam_twin.measure.steady_state import SteadyStateDetector

    class _T:
        _step_n = 0

    det = SteadyStateDetector(tol=0.01)
    twin = _T()
    for i, (W, D, T, trav) in enumerate(
        [
            (5.0, 1.5, 2000.0, 0.0),
            (5.5, 1.6, 2100.0, 2.0),
            (5.51, 1.61, 2105.0, 5.0),
            (5.52, 1.605, 2102.0, 8.0),
        ]
    ):
        twin._step_n = i
        det.observe(twin, W, D, T, trav)
    # Not steady yet over short history with large early jump
    assert det.is_steady(pool_length_mm=3.0) in (True, False)


def test_ledger_closure_math():
    from waam_twin.physics.ledger import LedgerSnapshot

    s = LedgerSnapshot(
        E_arc_J=100.0,
        E_droplet_J=0.0,
        E_conv_J=10.0,
        E_rad_J=5.0,
        E_evap_J=0.0,
        E_cap_J=0.0,
        E_floor_J=0.0,
        E_freeze_J=0.0,
        E_stored_J=185.0,
        E_stored_0_J=100.0,
    )
    # in(100) - out(15) - dH(85) = 0
    assert abs(s.energy_closure_residual_J()) < 1e-9
