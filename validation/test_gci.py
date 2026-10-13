"""Phase 6 — GCI math smoke (host-only)."""

from waam_twin.physics.gci import richardson_gci, study_invalid_if_clamps


def test_gci_monotonic_converging():
    # Synthetic second-order: f = f* + a h²
    f_star = 5.0
    a = 2.0
    hs = [0.1, 0.2, 0.4]
    fs = [f_star + a * h * h for h in hs]
    r = richardson_gci(hs, fs)
    assert r.valid
    assert abs(r.p - 2.0) < 0.3
    assert abs(r.f_exact - f_star) < 0.2
    assert r.gci_fine < 0.1


def test_clamps_invalidate():
    rows = [{"dx_mm": 0.3, "force_clamp_steps": 10, "mach_clamp_steps": 0}]
    reasons = study_invalid_if_clamps(rows)
    assert reasons
