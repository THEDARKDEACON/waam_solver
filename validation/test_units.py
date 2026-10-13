"""Unit conversion and dt-policy smoke tests (host-only)."""

from __future__ import annotations

import pytest

from waam_twin.units import LatticeScale, choose_dt, assert_no_lattice_job_keys


def test_force_scale_units():
    s = LatticeScale(dx=4e-4, dt=4e-5, rho=7800.0)
    # F_lu = F_phys * dt² / (ρ dx²)
    assert s.force_scale == pytest.approx((4e-5) ** 2 / (7800.0 * (4e-4) ** 2))


def test_tau_from_nu():
    s = LatticeScale(dx=4e-4, dt=4e-5, rho=7800.0)
    nu = 6.0e-7
    tau = s.tau_from_nu(nu)
    assert tau == pytest.approx(3.0 * nu * s.dt / s.dx ** 2 + 0.5)


def test_choose_dt_acoustic():
    # Fine mesh can satisfy both τ and Ma
    dt = choose_dt(1e-4, nu_phys=6e-7, u_design_m_s=1.0, ma_max=0.08, tau_min=0.5005)
    assert dt > 0
    # Coarse production dx cannot satisfy τ≥0.5005 and Ma≤0.08 together (C2)
    with pytest.raises(ValueError):
        choose_dt(4e-4, nu_phys=6e-7, u_design_m_s=1.0, ma_max=0.08, tau_min=0.5005)


def test_reject_lattice_job_keys():
    with pytest.raises(ValueError):
        assert_no_lattice_job_keys({"C_darcy": 1.6e5})


def test_typed_config_unknown_key():
    from waam_twin.config_types.schema import ThermalConfig, _merge_dataclass

    with pytest.raises(ValueError):
        _merge_dataclass(ThermalConfig, {"not_a_real_key": 1})
