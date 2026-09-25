"""
runtime.py — Backend detection, presets, and grid sizing (waam_twin v2)
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

from .compiler import ti

from .lattice import estimate_fields_vram_mb

from .paths import PROJECT_ROOT as _PROJECT_ROOT

_PRESETS_PATH = _PROJECT_ROOT / "config" / "presets.yaml"

_taichi_initialized = False
_profile: "PlatformProfile | None" = None


@dataclass
class PlatformProfile:
    backend: str
    device_name: str
    vram_mb: int | None
    ram_mb: int
    tier: str


@dataclass
class PresetConfig:
    """Hardware / cost profile — not the weld coupon geometry.

    ``vram_budget_mb`` and ``max_cells`` limit *cell count*, which is the
    practical compute-intensity knob (timestep work ∝ N_cells × physics_tier).
    Target for dx coarsening is that cell/memory budget — **not** viewer FPS.
    """

    name: str
    vram_budget_mb: int
    target_dx_mm: float
    max_tracers: int
    use_srt: bool
    max_cells: int | None = None


# Demo-only fallback when ``from_preset`` is called with no domain and no job.
DEMO_DEFAULT_DOMAIN_MM: tuple[float, float, float] = (80.0, 40.0, 25.0)


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for waam_twin v2 presets. Install with: pip install pyyaml"
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f)


def load_presets() -> dict[str, PresetConfig]:
    if not _PRESETS_PATH.exists():
        raise FileNotFoundError(f"Presets file not found: {_PRESETS_PATH}")
    raw = _load_yaml(_PRESETS_PATH)
    out: dict[str, PresetConfig] = {}
    for name, cfg in raw.items():
        if "domain_mm" in cfg:
            from . import logging_util as log
            log.warning(
                f"[presets] '{name}.domain_mm' is ignored — geometry belongs in the "
                f"job YAML (simulation.domain_mm / plate.size_mm)."
            )
        max_cells = cfg.get("max_cells")
        out[name] = PresetConfig(
            name=name,
            vram_budget_mb=int(cfg.get("vram_budget_mb", 2048)),
            target_dx_mm=float(cfg.get("target_dx_mm", 0.3)),
            max_tracers=int(cfg.get("max_tracers", 20000)),
            use_srt=bool(cfg.get("use_srt", True)),
            max_cells=int(max_cells) if max_cells is not None else None,
        )
    return out


def resolve_grid_budget_mb(preset: PresetConfig) -> int:
    """Effective memory cap for ``auto_grid``.

    Precedence:
      1. ``WAAM_VRAM_MB`` — explicit override (use to raise/lower the cap)
      2. preset ``vram_budget_mb`` — hardware-profile default

    Detected device VRAM is **not** used as the budget: filling an 8 GB card
    would make ``minimal`` as heavy as ``high``. Empty device memory means the
    profile's compute cap is doing its job.
    """
    override = os.environ.get("WAAM_VRAM_MB")
    if override:
        return max(64, int(override))
    return int(preset.vram_budget_mb)

def resolve_preset(name: str | None = None) -> PresetConfig:
    env = os.environ.get("WAAM_PRESET", "standard")
    key = (name or env).lower()
    if key == "auto":
        key = _auto_tier_from_vram()
    presets = load_presets()
    if key not in presets:
        raise KeyError(f"Unknown preset '{key}'. Available: {list(presets.keys())}")
    return presets[key]


def _auto_tier_from_vram() -> str:
    vram = _detect_vram_mb()
    if vram is None or vram < 1024:
        return "minimal"
    if vram < 6144:
        return "standard"
    if vram < 12288:
        return "high"
    return "ultra"


def _detect_vram_mb() -> int | None:
    override = os.environ.get("WAAM_VRAM_MB")
    if override:
        return int(override)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        # CSV rows look like: device,VRAM Total Memory (B),VRAM Total Used Memory (B)
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            return max(1, int(parts[1]) // (1024 * 1024))
    except Exception:
        return None
    return None


def _detect_ram_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 8192


def _runtime_is_live() -> bool:
    try:
        return ti.lang.impl.get_runtime().prog is not None
    except Exception:
        return False


# Named Quadrants arches this twin will try. ``python`` is opt-in (debugger).
BACKENDS: tuple[str, ...] = ("cpu", "cuda", "amdgpu", "vulkan", "metal", "python")
_AUTO_ORDER: tuple[tuple[str, object], ...] = (
    ("cuda", ti.cuda),
    ("amdgpu", ti.amdgpu),
    ("vulkan", ti.vulkan),
    ("cpu", ti.cpu),
)
_ARCH_BY_NAME = {
    "cpu": ti.cpu,
    "cuda": ti.cuda,
    "amdgpu": ti.amdgpu,
    "vulkan": ti.vulkan,
    "metal": ti.metal,
    "python": ti.python,
}


def init_taichi(backend: str | None = None) -> PlatformProfile:
    """Initialize the Quadrants runtime once.

    Auto order is native GPU first (CUDA, then ROCm/HIP), then Vulkan, then CPU.
    Fastcache and GPU graphs are not enabled here.

    Backend selection priority:
      1. ``WAAM_FORCE_BACKEND`` — sticky override (survives tests that mutate
         ``WAAM_BACKEND``, e.g. backend_smoke)
      2. ``WAAM_BACKEND`` — overrides hardcoded ``backend="cpu"`` in tests
      3. ``backend`` argument / ``auto``
    """
    global _taichi_initialized, _profile
    if _taichi_initialized and not _runtime_is_live():
        _taichi_initialized = False
        _profile = None

    allowed = set(BACKENDS) | {"auto"}
    force = (os.environ.get("WAAM_FORCE_BACKEND") or "").strip().lower()
    env_backend = (os.environ.get("WAAM_BACKEND") or "").strip().lower()
    if force in allowed:
        requested = force
    elif env_backend in allowed:
        requested = env_backend
    else:
        requested = (backend or "auto").lower()
        if requested not in allowed:
            requested = "auto"

    # Re-init if caller/env asks for a different device than the live runtime.
    if _taichi_initialized and _profile is not None:
        if requested != "auto" and _profile.backend != requested:
            reset_taichi()
        else:
            return _profile

    arch = None
    backend_used = "cpu"

    if requested in _ARCH_BY_NAME:
        arch = _ARCH_BY_NAME[requested]
        backend_used = requested
        ti.init(arch=arch, log_level=ti.WARN)
    else:
        for name, candidate in _AUTO_ORDER:
            try:
                ti.init(arch=candidate, log_level=ti.WARN)
                arch = candidate
                backend_used = name
                break
            except Exception:
                continue
        if arch is None:
            ti.init(arch=ti.cpu, log_level=ti.WARN)
            backend_used = "cpu"

    _taichi_initialized = True
    tier = resolve_preset().name if os.environ.get("WAAM_PRESET") else _auto_tier_from_vram()
    _profile = PlatformProfile(
        backend=backend_used,
        device_name=backend_used,
        vram_mb=_detect_vram_mb(),
        ram_mb=_detect_ram_mb(),
        tier=tier,
    )
    from . import logging_util as log
    log.info(
        f"[waam_twin] Backend={_profile.backend}  compiler=quadrants  "
        f"tier={_profile.tier}  vram={_profile.vram_mb}MB  ram={_profile.ram_mb}MB"
    )
    return _profile


def ensure_taichi() -> PlatformProfile:
    if not _taichi_initialized:
        return init_taichi()
    return _profile  # type: ignore[return-value]


def reset_taichi() -> None:
    """Reset the Quadrants runtime and clear the cached backend/profile state."""
    global _taichi_initialized, _profile
    try:
        ti.reset()
    finally:
        _taichi_initialized = False
        _profile = None


def auto_tracer_count(vram_mb: int | None, preset: PresetConfig) -> int:
    if vram_mb is None:
        return preset.max_tracers
    if vram_mb < 1024:
        return min(preset.max_tracers, 5000)
    if vram_mb < 4096:
        return min(preset.max_tracers, 20000)
    return preset.max_tracers


def vram_flags_from_job(job: dict | None) -> tuple[bool, bool, bool]:
    """Lorentz / VOF / export flags for auto_grid.

    Size against the **full optional set** when ``physics_tier`` is ``full``
    or Lorentz/VOF will be allocated (flags applied after grid construction).
    """
    sim = (job or {}).get("simulation") or {}
    tier = str(sim.get("physics_tier") or "flow").strip().lower()
    lorentz = bool(sim["enable_lorentz"]) if "enable_lorentz" in sim else tier == "full"
    vof = bool(sim["enable_vof"]) if "enable_vof" in sim else tier != "thermal"
    if tier == "full" or lorentz or vof:
        return True, True, True
    return bool(lorentz), bool(vof), True


def estimate_grid_vram_mb(
    nx: int,
    ny: int,
    nz: int,
    max_tracers: int,
    *,
    lorentz: bool = True,
    vof: bool = True,
    export: bool = True,
) -> float:
    """VRAM estimate matching WAAMGrid (defaults to full optional field set)."""
    return estimate_fields_vram_mb(
        nx, ny, nz, max_tracers,
        lorentz=lorentz, vof=vof, export=export,
    )


def grid_at_dx(
    domain_mm: tuple[float, float, float],
    dx_mm: float,
) -> tuple[int, int, int, float]:
    """``(nx, ny, nz, dx_m)`` for a fixed domain and cell size. No coarsening."""
    if dx_mm <= 0.0:
        raise ValueError(f"dx_mm must be > 0, got {dx_mm}")
    dx_m = float(dx_mm) / 1000.0
    lx, ly, lz = (float(d) / 1000.0 for d in domain_mm)
    nx = max(8, int(lx / dx_m))
    ny = max(8, int(ly / dx_m))
    nz = max(8, int(lz / dx_m))
    return nx, ny, nz, dx_m


def resolve_grid(
    domain_mm: tuple[float, float, float],
    dx_mm: float,
    vram_budget_mb: int,
    max_tracers: int = 20000,
    max_cells: int | None = None,
    *,
    auto_grid_enabled: bool = True,
    lorentz: bool = True,
    vof: bool = True,
    export: bool = True,
) -> tuple[int, int, int, float]:
    """Fit ``dx`` to the preset budget, or keep the requested ``dx`` when off.

    Preset collision model and tracer count are chosen by the caller either way.
    This only controls whether ``dx`` may be coarsened to ``max_cells`` / VRAM.
    """
    if auto_grid_enabled:
        return auto_grid(
            domain_mm, dx_mm, vram_budget_mb, max_tracers, max_cells=max_cells,
            lorentz=lorentz, vof=vof, export=export,
        )

    from . import logging_util as log

    nx, ny, nz, dx_m = grid_at_dx(domain_mm, dx_mm)
    est = estimate_grid_vram_mb(
        nx, ny, nz, max_tracers, lorentz=lorentz, vof=vof, export=export,
    )
    n_cells = nx * ny * nz
    budget = float(vram_budget_mb) * 0.85
    over = est > budget or (max_cells is not None and n_cells > max_cells)
    msg = (
        f"[auto_grid] off — requested dx={float(dx_mm):.3f} mm kept "
        f"(grid {nx}×{ny}×{nz}, ~{est:.0f} MB, {n_cells} cells). "
        "Preset VRAM/max_cells cap is not applied."
    )
    if over:
        log.warning(
            msg + " This grid is over the preset budget and may run out of memory."
        )
    else:
        log.info(msg)
    return nx, ny, nz, dx_m


def auto_grid(
    domain_mm: tuple[float, float, float],
    target_dx_mm: float,
    vram_budget_mb: int,
    max_tracers: int = 20000,
    max_cells: int | None = None,
    *,
    log_coarsen: bool = True,
    lorentz: bool = True,
    vof: bool = True,
    export: bool = True,
) -> tuple[int, int, int, float]:
    """
    Pick (nx, ny, nz, dx) for a fixed physical domain.

    Domain size is never shrunk. If the requested ``target_dx_mm`` needs more
    memory or cells than the hardware profile allows, ``dx`` is coarsened
    until the grid fits (or a MemoryError is raised).
    """
    from . import logging_util as log

    dx_m = target_dx_mm / 1000.0
    dx_req_mm = target_dx_mm
    lx, ly, lz = (d / 1000.0 for d in domain_mm)

    nx = max(8, int(lx / dx_m))
    ny = max(8, int(ly / dx_m))
    nz = max(8, int(lz / dx_m))

    budget = float(vram_budget_mb) * 0.85  # headroom for Taichi runtime
    cell_cap = int(max_cells) if max_cells is not None else None

    def _over_budget() -> bool:
        if estimate_grid_vram_mb(
            nx, ny, nz, max_tracers, lorentz=lorentz, vof=vof, export=export,
        ) > budget:
            return True
        if cell_cap is not None and nx * ny * nz > cell_cap:
            return True
        return False

    while _over_budget() and dx_m < 0.002:
        dx_m *= 1.15
        nx = max(8, int(lx / dx_m))
        ny = max(8, int(ly / dx_m))
        nz = max(8, int(lz / dx_m))

    est = estimate_grid_vram_mb(
        nx, ny, nz, max_tracers, lorentz=lorentz, vof=vof, export=export,
    )
    n_cells = nx * ny * nz
    if est > budget or (cell_cap is not None and n_cells > cell_cap):
        raise MemoryError(
            f"Grid {nx}×{ny}×{nz} (~{est:.0f} MB, {n_cells} cells) exceeds hardware "
            f"budget {vram_budget_mb} MB"
            + (f" / max_cells={cell_cap}" if cell_cap is not None else "")
            + ". Raise WAAM_VRAM_MB, use a larger hardware profile, or shrink "
            "simulation.domain_mm in the job."
        )

    dx_mm = dx_m * 1000.0
    if log_coarsen and dx_mm > dx_req_mm * 1.02:
        reason = []
        if cell_cap is not None:
            reason.append(f"max_cells={cell_cap}")
        reason.append(f"vram_budget≈{vram_budget_mb} MB")
        log.info(
            f"[auto_grid] Kept domain {domain_mm[0]:.0f}×{domain_mm[1]:.0f}×{domain_mm[2]:.0f} mm; "
            f"coarsened dx {dx_req_mm:.3f}→{dx_mm:.3f} mm to fit ({', '.join(reason)}). "
            f"Grid {nx}×{ny}×{nz} (~{est:.0f} MB). Target is cell/VRAM budget, not FPS."
        )

    return nx, ny, nz, dx_m


def check_vram_budget(
    nx: int,
    ny: int,
    nz: int,
    max_tracers: int,
    budget_mb: int,
    *,
    lorentz: bool = True,
    vof: bool = True,
    export: bool = True,
) -> None:
    est = estimate_grid_vram_mb(
        nx, ny, nz, max_tracers, lorentz=lorentz, vof=vof, export=export,
    )
    if est > budget_mb:
        raise MemoryError(
            f"Estimated VRAM {est:.1f} MB exceeds budget {budget_mb} MB. "
            f"Try a coarser dx, WAAM_VRAM_MB, or a smaller simulation.domain_mm."
        )
