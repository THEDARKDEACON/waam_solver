"""
Energy and mass ledger — GPU-tracked conservation bookkeeping.

Terms (SI):
  inputs:  arc heat, droplet enthalpy
  outputs: convection, radiation, evaporation, domain-face conduction
  corrections: enthalpy-cap, enthalpy-floor, freeze adjustments
  storage: ΔH stored in the metal
  mass: wire mass fed vs metal mass present above the plate
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..twin import WAAMTwin


@dataclass
class LedgerSnapshot:
    """Host-side ledger view in joules / kilograms."""

    step: int = 0
    sim_time_s: float = 0.0
    # Energy [J]
    E_arc_J: float = 0.0
    E_droplet_J: float = 0.0
    E_conv_J: float = 0.0
    E_rad_J: float = 0.0
    E_evap_J: float = 0.0
    E_cap_J: float = 0.0
    E_floor_J: float = 0.0
    E_freeze_J: float = 0.0
    # Signed enthalpy carried into the metal by gas<->metal relabelling.
    E_relabel_J: float = 0.0
    E_stored_J: float = 0.0
    E_stored_0_J: float = 0.0
    # Mass [kg]
    m_wire_kg: float = 0.0
    m_deposited_kg: float = 0.0
    m_phi_above_kg: float = 0.0

    def energy_in_J(self) -> float:
        return self.E_arc_J + self.E_droplet_J

    def energy_out_J(self) -> float:
        return (
            self.E_conv_J
            + self.E_rad_J
            + self.E_evap_J
            + self.E_cap_J
            + self.E_freeze_J
        )

    def energy_closure_residual_J(self) -> float:
        """in − out − (stored − stored0) + floor restore + relabel. Ideal = 0."""
        dH = self.E_stored_J - self.E_stored_0_J
        return (
            self.energy_in_J() - self.energy_out_J() - dH
            + self.E_floor_J + self.E_relabel_J
        )

    def energy_closure_rel(self) -> float:
        scale = max(abs(self.energy_in_J()), abs(self.E_stored_0_J), 1.0)
        return abs(self.energy_closure_residual_J()) / scale

    def mass_balance_ratio(self) -> float:
        return self.m_deposited_kg / max(self.m_wire_kg, 1e-30)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["energy_closure_rel"] = self.energy_closure_rel()
        d["energy_residual_J"] = self.energy_closure_residual_J()
        d["mass_balance_ratio"] = self.mass_balance_ratio()
        return d


@dataclass
class EnergyMassLedger:
    """Accumulators attached to a twin; updated from coupled_step telemetry."""

    E_arc_J: float = 0.0
    E_droplet_J: float = 0.0
    E_conv_J: float = 0.0
    E_rad_J: float = 0.0
    E_evap_J: float = 0.0
    E_cap_J: float = 0.0
    E_floor_J: float = 0.0
    E_freeze_J: float = 0.0
    E_stored_0_J: float | None = None
    _last_stored_J: float = 0.0
    _acc_base_J: tuple[float, float, float, float] | None = None
    notes: list[str] = field(default_factory=list)

    def attach(self, twin: "WAAMTwin") -> None:
        twin.ledger = self  # type: ignore[attr-defined]
        self._acc_base_J = _device_acc_J(twin)

    def _device_since_attach_J(self, twin: "WAAMTwin") -> tuple[float, float, float, float]:
        """Convection, radiation, floor and relabel joules booked on device since attach."""
        now = _device_acc_J(twin)
        if self._acc_base_J is None:
            self._acc_base_J = now
        return tuple(n - b for n, b in zip(now, self._acc_base_J))  # type: ignore[return-value]

    def record_arc(self, joules: float) -> None:
        self.E_arc_J += max(0.0, float(joules))

    def record_droplet_enthalpy(self, joules: float) -> None:
        self.E_droplet_J += float(joules)

    def record_evap(self, joules: float) -> None:
        self.E_evap_J += max(0.0, float(joules))

    def record_cap(self, joules: float) -> None:
        self.E_cap_J += max(0.0, float(joules))

    def record_floor(self, joules: float) -> None:
        self.E_floor_J += max(0.0, float(joules))

    def record_freeze(self, joules: float) -> None:
        self.E_freeze_J += max(0.0, float(joules))

    def record_losses(self, conv_J: float = 0.0, rad_J: float = 0.0) -> None:
        self.E_conv_J += max(0.0, float(conv_J))
        self.E_rad_J += max(0.0, float(rad_J))

    def snapshot(self, twin: "WAAMTwin") -> LedgerSnapshot:
        g = twin.grid
        stored = _stored_enthalpy_J(twin)
        if self.E_stored_0_J is None:
            self.E_stored_0_J = stored
        self._last_stored_J = stored

        from .deposition_balance import expected_deposited_mass_kg

        m_wire = float(expected_deposited_mass_kg(twin))
        m_dep = float(getattr(twin, "_deposited_volume_m3", 0.0)) * float(twin.mat.rho)
        m_phi = _phi_metal_above_kg(twin)
        conv_J, rad_J, floor_J, relabel_J = self._device_since_attach_J(twin)

        return LedgerSnapshot(
            step=int(getattr(twin, "_step_n", 0)),
            sim_time_s=float(getattr(twin, "_step_n", 0)) * float(g.dt),
            E_arc_J=self.E_arc_J,
            E_droplet_J=self.E_droplet_J,
            E_conv_J=self.E_conv_J + conv_J,
            E_rad_J=self.E_rad_J + rad_J,
            E_evap_J=self.E_evap_J,
            E_cap_J=self.E_cap_J,
            E_floor_J=self.E_floor_J + floor_J,
            E_relabel_J=relabel_J,
            E_freeze_J=self.E_freeze_J,
            E_stored_J=stored,
            E_stored_0_J=float(self.E_stored_0_J),
            m_wire_kg=m_wire,
            m_deposited_kg=m_dep,
            m_phi_above_kg=m_phi,
        )


def _device_acc_J(twin: "WAAMTwin") -> tuple[float, float, float, float]:
    acc = getattr(twin.grid, "energy_acc_J", None)
    if acc is None:
        return 0.0, 0.0, 0.0, 0.0
    v = acc.to_numpy()
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


def _stored_enthalpy_J(twin: "WAAMTwin") -> float:
    """Sum w·H·Δx³ over non-gas cells [J] (w = fill under the free-surface LBM)."""
    import numpy as np

    g = twin.grid
    H = g.H.to_numpy().astype(np.float64)
    flags = g.flags.to_numpy()
    metal = flags != g.FLAG_GAS
    if twin.fs_iface_flag() >= 0:
        H = H * np.clip(g.phi.to_numpy(), 0.0, 1.0)
    return float((H[metal]).sum() * (g.dx ** 3))


def _phi_metal_above_kg(twin: "WAAMTwin") -> float:
    import numpy as np

    g = twin.grid
    phi = g.phi.to_numpy()
    flags = g.flags.to_numpy()
    nz = int(getattr(twin, "nz_solid", 0))
    mask = flags != g.FLAG_GAS
    if nz > 0:
        mask[:, :, :nz] = False
    vol = float((phi * mask).sum() * (g.dx ** 3))
    return vol * float(twin.mat.rho)


def ensure_ledger(twin: "WAAMTwin") -> EnergyMassLedger:
    led = getattr(twin, "ledger", None)
    if led is None or not isinstance(led, EnergyMassLedger):
        led = EnergyMassLedger()
        led.attach(twin)
    return led
