"""Arc heat source models (waam_twin v2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .. import kernels
from .. import logging_util as log

if TYPE_CHECKING:
    from ..grid import WAAMGrid
    from ..twin import WAAMTwin


class ArcHeatSource(Protocol):
    def inject(
        self,
        twin: "WAAMTwin",
        g: "WAAMGrid",
        arc_i: float,
        arc_j: float,
        arc_k: float,
    ) -> None: ...


def _arc_weight_args(twin: "WAAMTwin") -> tuple[float, int]:
    pen_cells = twin.arc_penetration_m / twin.grid.dx
    enable = 1 if twin.arc_surface_weighting else 0
    return pen_cells, enable


def normalize_goldak_fractions(ff: float, fr: float) -> tuple[float, float]:
    """
    Enforce Goldak f_f + f_r = 2.

    Fractions that sum to ~1 (common YAML mistake) are scaled ×2.
    Other sums are renormalized to 2 with a warning.
    """
    s = float(ff) + float(fr)
    if s <= 1e-12:
        log.warning("[Goldak] ff+fr≤0; using defaults ff=0.6, fr=1.4")
        return 0.6, 1.4
    if abs(s - 2.0) <= 0.05:
        return float(ff), float(fr)
    if abs(s - 1.0) <= 0.05:
        log.warning(
            f"[Goldak] ff+fr={s:.3f} (unity fractions); scaling ×2 → "
            f"ff={2.0 * ff:.3f}, fr={2.0 * fr:.3f}"
        )
        return 2.0 * float(ff), 2.0 * float(fr)
    log.warning(
        f"[Goldak] ff+fr={s:.3f} ≠ 2; renormalizing to sum 2 "
        f"(ff={2.0 * ff / s:.3f}, fr={2.0 * fr / s:.3f})"
    )
    return 2.0 * float(ff) / s, 2.0 * float(fr) / s


def _arc_power_W(twin: "WAAMTwin") -> tuple[float, float]:
    """
    Phase 4 energy split: return (Q_inject, eta_inject) for the arc surface
    source.

    When ``twin.energy_split`` is True, Q_arc = η·Q − Q_droplet_booked, where
    Q_droplet_booked is the enthalpy *actually* deposited by gas→fluid
    conversion on the previous step (not the theoretical wire power). Subtracting
    the theoretical droplet power after removing deposition pass-3 starved the
    pool (~600 W arc vs ~1600 W ηQ).
    """
    if not getattr(twin, "energy_split", False):
        return float(twin.Q_w), float(twin.eta)
    from .heat_source import energy_split_W
    split = energy_split_W(twin)
    twin._last_energy_split = split
    dt = max(float(twin.grid.dt), 1e-30)
    # Prefer previous-step booking (inject runs before this step's deposition).
    booked_J = float(
        getattr(twin, "_droplet_enthalpy_booked_J_prev", 0.0)
        or getattr(twin, "_droplet_enthalpy_booked_J_step", 0.0)
    )
    booked_W = booked_J / dt
    q_arc = max(0.0, float(split["Q_total_W"]) - booked_W)
    return q_arc, 1.0


@dataclass
class Gaussian2D:
    """Default surface Gaussian — production path."""

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        pen_cells, enable = _arc_weight_args(twin)
        Q, eta = _arc_power_W(twin)
        kernels.inject_arc_heat(
            g.H, g.flags, g.phi, g.f_l, g.arc_norm_buf,
            arc_i, arc_j, arc_k,
            Q, twin.sigma_cells,
            g.dt, g.dx ** 3, eta,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
            twin.fs_iface_flag(),
        )


@dataclass
class Goldak3D:
    """Goldak double-ellipsoid heat source (Goldak 1984).

    Semi-axes are in lattice cells. Values ≤0 are resolved at inject time
    from the twin's arc σ and legacy depth fields.

    normalize_energy:
      True  — metal-only rescale so ΣΔH·dx³ = ηQΔt (historical twin path)
      False — absolute CF like Ansys MAPDL (no rescale; ARG<9 cutoff)
    """

    ff: float = 0.6
    fr: float = 1.4
    a_front: float = 0.0
    a_rear: float = 0.0
    b: float = 0.0
    c: float = 0.0
    # Legacy / fallback when a,b,c not set from job mm fields
    depth_front: float = 2.0
    depth_rear: float = 4.0
    sigma_scale: float = 1.0
    normalize_energy: bool = True

    def _resolved_axes(self, twin: "WAAMTwin") -> tuple[float, float, float, float]:
        sig = max(0.5, twin.sigma_cells * self.sigma_scale)
        a_f = self.a_front if self.a_front > 0.0 else sig
        a_r = self.a_rear if self.a_rear > 0.0 else max(sig, 2.0 * sig)
        b = self.b if self.b > 0.0 else sig
        if self.c > 0.0:
            c = self.c
        else:
            # Prefer front depth; fall back to rear if only that was set.
            c = max(0.5, self.depth_front if self.depth_front > 0.0 else self.depth_rear)
        return a_f, a_r, b, c

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        # Align front/rear with torch travel in the XY plane (not ±grid-x alone).
        dir_x, dir_y, _ = getattr(twin, "_torch_dir_xyz", (1.0, 0.0, 0.0))
        horiz = (float(dir_x) ** 2 + float(dir_y) ** 2) ** 0.5
        if horiz < 1e-12:
            # Stationary / vertical tip motion: fall back to signed travel speed on +x.
            dir_x = 1.0 if twin.travel_speed_m_s >= 0.0 else -1.0
            dir_y = 0.0
        else:
            dir_x = float(dir_x) / horiz
            dir_y = float(dir_y) / horiz
        pen_cells, enable = _arc_weight_args(twin)
        a_f, a_r, b, c = self._resolved_axes(twin)
        Q, eta = _arc_power_W(twin)
        normalize = 1 if self.normalize_energy else 0
        kernels.inject_goldak_heat(
            g.H, g.flags, g.phi, g.f_l, g.arc_norm_buf,
            arc_i, arc_j, arc_k,
            Q,
            g.dt, g.dx ** 3, eta,
            dir_x, dir_y, self.ff, self.fr,
            a_f, a_r, b, c,
            pen_cells, enable,
            g.FLAG_SOLID, g.FLAG_GAS,
            normalize,
            twin.fs_iface_flag(),
        )
        if self.normalize_energy:
            twin._last_goldak_deposited_J = float(eta) * float(Q) * float(g.dt)
        else:
            twin._last_goldak_deposited_J = float(g.arc_norm_buf.to_numpy())


@dataclass
class ConicalVolume:
    """Conical volume heat source — not yet implemented (do not select in jobs)."""

    sigma_scale: float = 1.2

    def inject(self, twin: "WAAMTwin", g: "WAAMGrid", arc_i: float, arc_j: float, arc_k: float) -> None:
        raise NotImplementedError(
            "heat_source 'conical' is not implemented yet "
            "(was a widened-Gaussian stub). Use gaussian2d or goldak."
        )


def create_heat_source(name: str, **kwargs) -> ArcHeatSource:
    key = (name or "gaussian2d").lower().replace("-", "").replace("_", "")
    if key in ("goldak", "goldak3d", "doubleellipsoid"):
        ff, fr = normalize_goldak_fractions(
            float(kwargs.pop("ff", 0.6)),
            float(kwargs.pop("fr", 1.4)),
        )
        return Goldak3D(ff=ff, fr=fr, **kwargs)
    if key in ("conical", "conicalvolume", "cone"):
        raise ValueError(
            "heat_source 'conical' is not implemented. "
            "Valid: gaussian2d | goldak (aliases: goldak3d, doubleellipsoid)"
        )
    if key in ("gaussian2d", "gaussian", "gauss"):
        return Gaussian2D()
    raise ValueError(
        f"Unknown heat_source '{name}'. "
        "Valid: gaussian2d | goldak (aliases: goldak3d, doubleellipsoid)"
    )


def goldak_from_job_mm(twin: "WAAMTwin", goldak_cfg: dict) -> Goldak3D:
    """Build Goldak3D with semi-axes in cells from job mm fields.

    Preferred keys (PHYSICS_FORCE_CORRECTNESS_SPEC §5.4):
      a_front_mm, a_rear_mm, b_mm, c_mm, ff, fr  (ff+fr → 2)

    Legacy keys still accepted:
      depth_front_mm / depth_rear_mm → c (and depth_* cell fallbacks)
      sigma_scale → scales a,b from arc σ when a_*/b not given
    """
    dx = twin.grid.dx
    dx_mm = dx * 1000.0
    cfg = goldak_cfg or {}

    ff, fr = normalize_goldak_fractions(
        float(cfg.get("ff", 0.6)),
        float(cfg.get("fr", 1.4)),
    )
    sigma_scale = float(cfg.get("sigma_scale", 1.0))
    sig_cells = max(0.5, twin.sigma_cells * sigma_scale)

    def _mm_to_cells(key: str, default_cells: float | None = None) -> float:
        if key in cfg and cfg[key] is not None:
            return max(0.5, float(cfg[key]) / dx_mm)
        if default_cells is None:
            return 0.0
        return max(0.5, float(default_cells))

    # Explicit Goldak axes (0 ⇒ resolve at inject from σ / legacy depths)
    a_front = _mm_to_cells("a_front_mm", None)
    a_rear = _mm_to_cells("a_rear_mm", None)
    b = _mm_to_cells("b_mm", None)
    c = _mm_to_cells("c_mm", None)

    depth_front = _mm_to_cells("depth_front_mm", 2.0)
    depth_rear = _mm_to_cells("depth_rear_mm", 4.0)

    # If only legacy depths given, map: a~σ, b~σ, c~depth_front
    if a_front <= 0.0 and a_rear <= 0.0 and b <= 0.0 and c <= 0.0:
        a_front = sig_cells
        a_rear = max(sig_cells, 2.0 * sig_cells)
        b = sig_cells
        c = depth_front
        log.info(
            f"[Goldak] legacy depths → a_f={a_front:.2f} a_r={a_rear:.2f} "
            f"b={b:.2f} c={c:.2f} cells (σ_scale={sigma_scale})"
        )
    elif c <= 0.0:
        c = depth_front

    normalize_energy = bool(cfg.get("normalize_energy", True))
    if not normalize_energy:
        log.info(
            "[Goldak] normalize_energy=false — absolute CF (Ansys/APDL style), "
            "no metal-only rescale"
        )

    return Goldak3D(
        ff=ff,
        fr=fr,
        a_front=a_front,
        a_rear=a_rear,
        b=b,
        c=c,
        depth_front=depth_front,
        depth_rear=depth_rear,
        sigma_scale=sigma_scale,
        normalize_energy=normalize_energy,
    )


__all__ = [
    "ArcHeatSource",
    "Gaussian2D",
    "Goldak3D",
    "ConicalVolume",
    "create_heat_source",
    "goldak_from_job_mm",
    "normalize_goldak_fractions",
]
