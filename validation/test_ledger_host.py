"""Host-side ledger math (no GPU)."""

from waam_twin.physics.ledger import LedgerSnapshot


def test_closure_perfect():
    s = LedgerSnapshot(
        E_arc_J=50.0,
        E_droplet_J=10.0,
        E_conv_J=5.0,
        E_rad_J=3.0,
        E_evap_J=2.0,
        E_cap_J=0.0,
        E_floor_J=0.0,
        E_freeze_J=0.0,
        E_stored_J=150.0,
        E_stored_0_J=100.0,
    )
    assert abs(s.energy_closure_residual_J()) < 1e-9
    assert s.energy_closure_rel() < 1e-9


def test_mass_ratio():
    s = LedgerSnapshot(m_wire_kg=0.01, m_deposited_kg=0.0095)
    assert abs(s.mass_balance_ratio() - 0.95) < 1e-12
