"""Research export bundle: volume + surface + tracers + metadata."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Callable, Sequence

from .meta import write_meta_json
from .vtk_io import (
    TIER_CORE,
    TIER_DERIVED,
    TIER_FORCES,
    TIER_MATERIAL,
    export_surface,
    export_tracers,
    export_volume,
)

if TYPE_CHECKING:
    from ..twin import WAAMTwin

TIER_MAP = {
    0: TIER_CORE,
    1: TIER_MATERIAL,
    2: TIER_FORCES,
    3: TIER_DERIVED,
}


def _parse_tiers(tiers: Sequence[int]) -> tuple[int, ...]:
    return tuple(TIER_MAP.get(t, t) for t in tiers)


def export_research_bundle(
    twin: "WAAMTwin",
    out_dir: str | pathlib.Path,
    tag: str | None = None,
    tiers: Sequence[int] = (0, 1, 3),
    include_surface: bool = True,
    include_tracers: bool = True,
    include_sidecar: bool = True,
    crop_liquid: bool = False,
    job_path: str | None = None,
) -> dict[str, str]:
    """Write a complete research snapshot to *out_dir*."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tier_tuple = _parse_tiers(tiers)
    step_tag = tag or f"step_{twin._step_n:06d}"
    paths: dict[str, str] = {}

    vol_path = out_dir / f"volume_{step_tag}.vti"
    p = export_volume(twin, str(vol_path), tiers=tier_tuple, crop_liquid=crop_liquid)
    if p:
        paths["volume"] = p

    if include_surface and twin.enable_vof:
        surf_path = out_dir / f"surface_{step_tag}.vtp"
        p = export_surface(twin, str(surf_path))
        if p:
            paths["surface"] = p

    if include_tracers:
        tr_path = out_dir / f"tracers_{step_tag}.vtp"
        p = export_tracers(twin, str(tr_path))
        if p:
            paths["tracers"] = p

    telem_path = out_dir / f"telemetry_{step_tag}.json"
    with open(telem_path, "w") as f:
        json.dump(twin.get_telemetry(), f, indent=2)
    paths["telemetry"] = str(telem_path)

    if include_sidecar:
        meta_path = out_dir / f"meta_{step_tag}.json"
        write_meta_json(twin, meta_path, job_path=job_path)
        paths["meta"] = str(meta_path)

    if twin.probe_recorder is not None and twin.probe_recorder.probes:
        probe_path = out_dir / "probes.csv"
        twin.probe_recorder.write_csv(probe_path)
        paths["probes"] = str(probe_path)

    print(f"[export] Research bundle → {out_dir.resolve()}  ({len(paths)} files)")
    return paths


def write_pvd(
    collection_path: str | pathlib.Path,
    vti_paths: list[str],
    times_s: list[float] | None = None,
) -> None:
    """Write a ParaView PVD file referencing time-series VTI snapshots.

    Paths in the PVD are relative to the PVD file's directory (VTK convention).
    ``times_s`` gives the physical simulation time [s] of each snapshot; if
    omitted the frame index is used (ParaView's time axis then shows frame
    count, not seconds).
    """
    collection_path = pathlib.Path(collection_path).resolve()
    pvd_dir = collection_path.parent
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    for i, vti in enumerate(vti_paths):
        vti_p = pathlib.Path(vti)
        if not vti_p.is_absolute():
            vti_p = (pathlib.Path.cwd() / vti_p).resolve()
        else:
            vti_p = vti_p.resolve()
        try:
            rel = vti_p.relative_to(pvd_dir)
        except ValueError:
            rel = pathlib.Path(vti_p.name)
        t = times_s[i] if times_s is not None and i < len(times_s) else float(i)
        lines.append(f'    <DataSet timestep="{t:.9g}" file="{rel.as_posix()}"/>')
    lines.extend(["  </Collection>", "</VTKFile>"])
    collection_path.write_text("\n".join(lines))
    print(f"[export] PVD collection → {collection_path}")


def export_research_sequence(
    twin: "WAAMTwin",
    out_dir: str | pathlib.Path,
    n_steps: int,
    every_n: int = 100,
    torch_x_m: float = 0.01,
    torch_y_m: float | None = None,
    is_welding: bool = True,
    max_frames: int = 200,
    tiers: Sequence[int] = (0, 3),
    job_path: str | None = None,
    after_frame: Callable[[int, pathlib.Path, dict[str, str]], None] | None = None,
) -> list[str]:
    """Run simulation and export bundles on a schedule; write PVD at end."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = twin.grid
    if torch_y_m is None:
        torch_y_m = (g.ny // 2) * g.dx

    vti_paths: list[str] = []
    frame_times_s: list[float] = []
    frame = 0
    for step_i in range(n_steps):
        twin.step(torch_x_m, torch_y_m, is_welding=is_welding)
        if step_i % every_n != 0 and step_i != n_steps - 1:
            continue
        if frame >= max_frames:
            break
        tag = f"step_{twin._step_n:06d}"
        sub = out_dir / f"frame_{frame:04d}"
        paths = export_research_bundle(
            twin,
            sub,
            tag=tag,
            tiers=tiers,
            include_surface=True,
            include_tracers=(frame == 0 or frame % 5 == 0),
            job_path=job_path,
        )
        if after_frame is not None:
            after_frame(frame, sub, paths)
        if "volume" in paths:
            vti_paths.append(paths["volume"])
            frame_times_s.append(twin._step_n * g.dt)
        frame += 1

    if vti_paths:
        write_pvd(out_dir / "sequence.pvd", vti_paths, times_s=frame_times_s)
    return vti_paths
