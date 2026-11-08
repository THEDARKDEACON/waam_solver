"""
test_blowup_guard.py — a NaN/inf field is reported, not hidden.

Peak-temperature reductions skip NaN and the solver's fast-math folds
``x != x`` to false, so a blown-up run used to look like a cold plate. The
periodic health check (nan_check_every) must count the bad cells with a
bit test, record the step once, raise under strict_mode, expose it in
telemetry, and clear on reset.
"""

from __future__ import annotations

from waam_twin import WAAMTwin
from waam_twin.kernels.health import count_nonfinite
from waam_twin.runtime import init_taichi


def run() -> None:
    init_taichi()
    twin = WAAMTwin(nx=16, ny=12, nz=12, dx=4e-4, max_tracers=4, enable_vof=True)
    twin.reset()
    g = twin.grid
    assert int(count_nonfinite(g.T, g.H)) == 0
    twin._step_n = 200
    twin._check_health()
    assert twin._blown_up_step is None

    g.T[5, 5, 1] = float("nan")
    g.H[6, 5, 1] = float("-inf")
    assert int(count_nonfinite(g.T, g.H)) == 2
    twin._check_health()
    assert twin._blown_up_step == 200 and twin._nonfinite_cells == 2
    tel = twin.get_telemetry(force_diagnostics=False)
    assert tel["blown_up_step"] == 200 and tel["nonfinite_cells"] == 2

    twin._step_n = 400
    twin._check_health()
    assert twin._blown_up_step == 200  # first occurrence kept

    twin.reset()
    assert twin._blown_up_step is None and int(count_nonfinite(g.T, g.H)) == 0

    twin.strict_mode = True
    g.T[5, 5, 1] = float("nan")
    twin._step_n = 200
    try:
        twin._check_health()
    except RuntimeError as exc:
        assert "BLOW-UP" in str(exc)
    else:
        raise AssertionError("strict_mode did not stop on NaN")
    print("[blowup_guard] NaN/inf detected, reported once, strict raises, reset clears")


if __name__ == "__main__":
    run()
    print("PASS")
