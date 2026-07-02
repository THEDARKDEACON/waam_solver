"""
platform.py — Backend detection, presets, and grid sizing (waam_twin v2)
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

import taichi as ti

# Bytes per cell (approx): 2×19×4 dist + 7×4 scalar + 6×4 vector + flags
_BYTES_PER_CELL = 2 * 19 * 4 + 7 * 4 + 6 * 4 + 4

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
    name: str
    vram_budget_mb: int
    domain_mm: tuple[float, float, float]
    target_dx_mm: float
    max_tracers: int
    use_srt: bool


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
        domain = cfg.get("domain_mm", [80, 40, 25])
        out[name] = PresetConfig(
            name=name,
            vram_budget_mb=int(cfg.get("vram_budget_mb", 2048)),
            domain_mm=(float(domain[0]), float(domain[1]), float(domain[2])),
            target_dx_mm=float(cfg.get("target_dx_mm", 0.3)),
            max_tracers=int(cfg.get("max_tracers", 20000)),
            use_srt=bool(cfg.get("use_srt", True)),
        )
    return out


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


def init_taichi(backend: str | None = None) -> PlatformProfile:
    """Initialize Taichi once: CUDA → Vulkan → CPU."""
    global _taichi_initialized, _profile
    if _taichi_initialized and not _runtime_is_live():
        _taichi_initialized = False
        _profile = None
    if _taichi_initialized and _profile is not None:
        return _profile

    requested = (backend or os.environ.get("WAAM_BACKEND", "auto")).lower()
    arch = None
    backend_used = "cpu"

    if requested == "cpu":
        arch = ti.cpu
        backend_used = "cpu"
    elif requested == "cuda":
        arch = ti.cuda
        backend_used = "cuda"
    elif requested == "vulkan":
        arch = ti.vulkan
        backend_used = "vulkan"
    else:
        for candidate, name in ((ti.cuda, "cuda"), (ti.vulkan, "vulkan"), (ti.cpu, "cpu")):
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
        print(
            f"[waam_twin] Backend={_profile.backend}  tier={_profile.tier}  "
            f"vram={_profile.vram_mb}MB  ram={_profile.ram_mb}MB"
        )
        return _profile

    ti.init(arch=arch, log_level=ti.WARN)
    _taichi_initialized = True
    tier = _auto_tier_from_vram()
    _profile = PlatformProfile(
        backend=backend_used,
        device_name=backend_used,
        vram_mb=_detect_vram_mb(),
        ram_mb=_detect_ram_mb(),
        tier=tier,
    )
    print(
        f"[waam_twin] Backend={_profile.backend}  tier={_profile.tier}  "
        f"vram={_profile.vram_mb}MB  ram={_profile.ram_mb}MB"
    )
    return _profile


def ensure_taichi() -> PlatformProfile:
    if not _taichi_initialized:
        return init_taichi()
    return _profile  # type: ignore[return-value]


def reset_taichi() -> None:
    """Reset Taichi and clear the cached backend/profile state."""
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


def estimate_grid_vram_mb(nx: int, ny: int, nz: int, max_tracers: int) -> float:
    n = nx * ny * nz
    grid_bytes = n * _BYTES_PER_CELL
    tracer_bytes = max_tracers * (3 * 4 + 4)
    return (grid_bytes + tracer_bytes) / (1024 ** 2)


def auto_grid(
    domain_mm: tuple[float, float, float],
    target_dx_mm: float,
    vram_budget_mb: int,
    max_tracers: int = 20000,
) -> tuple[int, int, int, float]:
    """
    Pick (nx, ny, nz, dx) to fit domain and VRAM budget.
    dx may be coarsened if the requested resolution does not fit.
    """
    dx_m = target_dx_mm / 1000.0
    lx, ly, lz = (d / 1000.0 for d in domain_mm)

    nx = max(8, int(lx / dx_m))
    ny = max(8, int(ly / dx_m))
    nz = max(8, int(lz / dx_m))

    budget = float(vram_budget_mb) * 0.85  # headroom for Taichi runtime

    while estimate_grid_vram_mb(nx, ny, nz, max_tracers) > budget and dx_m < 0.002:
        dx_m *= 1.15
        nx = max(8, int(lx / dx_m))
        ny = max(8, int(ly / dx_m))
        nz = max(8, int(lz / dx_m))

    est = estimate_grid_vram_mb(nx, ny, nz, max_tracers)
    if est > budget:
        raise MemoryError(
            f"Grid {nx}×{ny}×{nz} needs ~{est:.0f} MB but budget is {vram_budget_mb} MB. "
            f"Use WAAM_PRESET=minimal or set WAAM_VRAM_MB."
        )

    return nx, ny, nz, dx_m


def check_vram_budget(nx: int, ny: int, nz: int, max_tracers: int, budget_mb: int) -> None:
    est = estimate_grid_vram_mb(nx, ny, nz, max_tracers)
    if est > budget_mb:
        raise MemoryError(
            f"Estimated VRAM {est:.1f} MB exceeds budget {budget_mb} MB. "
            f"Try WAAMTwin.from_preset('minimal') or a coarser dx."
        )
