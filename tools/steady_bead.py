"""Steady bead cross-section for calibration checks.

Reproduces the shop protocol: strict mode off, Lorentz solved every 2.5e-4 s,
fusion W/D and phi-crown height on a slice window behind the torch.

Usage (from the repo root, PYTHONPATH set to it):

  python -m waam_twin.tools.steady_bead \\
      --job waam_twin/jobs/examples/bead_calibrate.yaml \\
      --time-s 2.5 --x0 22 --x1 26 --repeat 3 \\
      --out waam_twin/runs/steady/spread.jsonl

Overrides (eta, Goldak axes in mm, marangoni, contact) are applied on the twin
after it is built. dx and sulphur are written into a temporary job, because dx
rebuilds the grid and sulphur rebuilds the surface-tension table at material
load. Re-running the Sahoo table after construction does not reach the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import time
from typing import Any

import yaml

from waam_twin.job import load_job_config, parse_torch_path
from waam_twin.measure.steady_section import steady_cross_section
from waam_twin.paths import PROJECT_ROOT
from waam_twin.runtime import reset_taichi
from waam_twin.torch_path import TorchPathDriver

_STEADY = PROJECT_ROOT / "runs" / "steady"


def _expected_area_mm2(job: dict) -> float | None:
    """Deposited cross-section implied by wire feed and travel speed."""
    process = job.get("process") or {}
    try:
        diameter = float(process["wire_diameter_mm"])
        wire_mm_s = float(process["wire_feed_m_min"]) / 60.0 * 1000.0
        travel_mm_s = float(process["travel_speed_mm_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if travel_mm_s <= 0.0:
        return None
    return math.pi / 4.0 * diameter * diameter * wire_mm_s / travel_mm_s


def _path_limit_s(job: dict) -> float:
    """Weld time at which the torch reaches the last waypoint."""
    travel_mm_s = float((job.get("process") or {}).get("travel_speed_mm_s", 0.0))
    if travel_mm_s <= 0.0:
        return float("inf")
    driver = TorchPathDriver(parse_torch_path(job), travel_mm_s / 1000.0)
    return driver.total_length / (travel_mm_s / 1000.0)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_override_job(
    job_path: pathlib.Path,
    *,
    dx_mm: float | None,
    sulphur_ppm: float | None,
    dt_scale: float | None = None,
) -> pathlib.Path:
    """Copy the job (and, for sulphur, its material) under runs/steady/."""
    _STEADY.mkdir(parents=True, exist_ok=True)
    job = load_job_config(job_path)
    sim = job.setdefault("simulation", {})
    if dx_mm is not None:
        sim["dx_mm"] = float(dx_mm)
    if dt_scale is not None:
        sim["dt_scale"] = float(dt_scale)
    if sulphur_ppm is not None:
        src = pathlib.Path(str(job.get("material", "")))
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        mat = yaml.safe_load(src.read_text(encoding="utf-8"))
        mat.setdefault("surfactant", {})["sulphur_ppm"] = float(sulphur_ppm)
        mat_path = _STEADY / f"_mat_s{sulphur_ppm:g}.yaml"
        mat_path.write_text(yaml.safe_dump(mat, sort_keys=False), encoding="utf-8")
        job["material"] = str(mat_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    out = _STEADY / "_job_override.yaml"
    out.write_text(yaml.safe_dump(job, sort_keys=False), encoding="utf-8")
    return out


def _apply_twin_overrides(twin, args: argparse.Namespace) -> None:
    twin.strict_mode = False
    twin.lorentz_solve_every_s = float(args.lorentz_every_s)
    if args.eta is not None:
        twin.eta = float(args.eta)
    cells_per_mm = 1.0 / (twin.grid.dx * 1e3)
    src = twin.arc_source
    for name, value in (
        ("a_front", args.a_front),
        ("a_rear", args.a_rear),
        ("b", args.b),
        ("c", args.c),
    ):
        if value is not None:
            setattr(src, name, float(value) * cells_per_mm)
    if args.marangoni is not None:
        twin.marangoni_scale = float(args.marangoni)
    if args.contact is not None:
        twin.contact_angle_deg = float(args.contact)
        twin.theta_rad = math.radians(twin.contact_angle_deg)
    if args.arc_sigma is not None:
        # With explicit Goldak axes this does not change the heat source. It
        # sets the deposition footprint, the Lorentz current spread and the
        # tracer spread. Arc pressure stays on pressure_sigma_mm.
        twin.arc_sigma_m = float(args.arc_sigma) / 1000.0
        twin.sigma_cells = twin.arc_sigma_m / twin.grid.dx
    if args.cs is not None:
        twin.smagorinsky_cs = float(args.cs)
    if args.force_limit_m_s2 is not None:
        twin.force_limit_m_s2 = float(args.force_limit_m_s2)
    if args.force_limit_lu is not None:
        twin.force_limit_lu = float(args.force_limit_lu)
    if args.c_darcy is not None:
        twin.C_darcy = float(args.c_darcy)
    if getattr(args, "no_lorentz", False):
        twin.enable_lorentz = False


def _knobs(twin, n_steps: int) -> dict[str, Any]:
    src = twin.arc_source
    mm = twin.grid.dx * 1e3
    return {
        "eta": float(twin.eta),
        "a_front_mm": float(getattr(src, "a_front", 0.0)) * mm,
        "a_rear_mm": float(getattr(src, "a_rear", 0.0)) * mm,
        "b_mm": float(getattr(src, "b", 0.0)) * mm,
        "c_mm": float(getattr(src, "c", 0.0)) * mm,
        "marangoni_scale": float(twin.marangoni_scale),
        "contact_angle_deg": float(twin.contact_angle_deg),
        "sulphur_ppm": float(getattr(twin.mat, "sulphur_ppm", 0.0)),
        "arc_sigma_mm": float(twin.arc_sigma_m) * 1000.0,
        "smagorinsky_cs": float(getattr(twin, "smagorinsky_cs", 0.0) or 0.0),
        "force_limit_m_s2": float(getattr(twin, "force_limit_m_s2", 0.0) or 0.0),
        "force_limit_lu": float(getattr(twin, "force_limit_lu", 0.0) or 0.0),
        "C_darcy": float(getattr(twin, "C_darcy", 0.0) or 0.0),
        "C_darcy_eff": float(twin.effective_C_darcy()),
        "dx_mm": float(mm),
        "dt_s": float(twin.grid.dt),
        "dt_scale": float(twin.grid.dt_scale),
        "tau": float(twin.grid.tau),
        "u_ceiling_m_s": 0.08 * float(twin.grid.dx) / float(twin.grid.dt),
        "n_steps": int(n_steps),
        "lorentz_solve_every_s": float(twin.lorentz_solve_every_s),
        "enable_lorentz": bool(twin.enable_lorentz),
        "strict_mode": bool(twin.strict_mode),
    }


def run_once(args: argparse.Namespace, job_path: pathlib.Path) -> dict[str, Any]:
    from waam_twin import WAAMTwin
    from waam_twin.measure.pool_flow import mean_flow_stats, pool_flow_stats

    twin = WAAMTwin.from_job(job_path)
    _apply_twin_overrides(twin, args)
    twin.reset()
    g = twin.grid
    ceiling = 0.08 * float(g.dx) / float(g.dt)
    print(
        f"CEILING u={ceiling:.3f} m/s  dt={g.dt * 1e6:.2f} us  "
        f"dt_scale={g.dt_scale:.4f}  tau={g.tau:.4f}",
        flush=True,
    )
    if args.dt_scale is not None and ceiling < 1.0:
        raise SystemExit(
            f"speed ceiling {ceiling:.3f} m/s is below 1.0 m/s; "
            f"the viscosity floor undid the shorter step"
        )
    if args.hash_fields and args.dt_scale is not None and ceiling < 1.4:
        raise SystemExit(
            f"speed ceiling {ceiling:.3f} m/s is below 1.4 m/s; "
            f"aborting the short-step hash gate"
        )
    if args.time_s is not None:
        n_steps = max(1, int(round(float(args.time_s) / float(g.dt))))
    else:
        n_steps = int(args.steps)
    job = load_job_config(job_path)
    section_job = load_job_config(args.job)
    driver = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
    sample_every = float(args.sample_every_s)
    t_sample0 = max(0.0, n_steps * float(g.dt) - 1.0)
    next_sample = t_sample0
    samples = []
    t0 = time.time()
    for step, x, y, _z in driver.positions_for_steps(n_steps, g.dt):
        twin.step(x, y, is_welding=True)
        t_now = (step + 1) * float(g.dt)
        if t_now + 0.5 * float(g.dt) >= next_sample:
            samples.append(pool_flow_stats(twin, float(twin.smagorinsky_cs)))
            next_sample += sample_every
    wall = time.time() - t0
    section = steady_cross_section(
        twin, float(args.x0), float(args.x1), section_job.get("reference") or {},
        area_expected_mm2=_expected_area_mm2(section_job),
    )
    record = {
        "git": _git_hash(),
        "job": str(args.job),
        **_knobs(twin, n_steps),
        **section,
        "mach_steps": int(getattr(twin, "_mach_clamp_steps", 0)),
        "mach_frac": int(getattr(twin, "_mach_clamp_steps", 0)) / n_steps,
        "force_clamp_frac": int(getattr(twin, "_force_clamp_steps", 0)) / n_steps,
        **mean_flow_stats(samples),
        "n_flow_samples": len(samples),
        "wall_s": wall,
        "ms_per_step": wall / n_steps * 1e3,
    }
    if args.hash_fields:
        import hashlib
        digest = hashlib.sha256()
        for name in ("phi", "flags", "H"):
            digest.update(getattr(g, name).to_numpy().tobytes())
        record["field_hash"] = digest.hexdigest()
        print(
            f"HASH phi,flags,H sha256={record['field_hash']}  steps={n_steps}",
            flush=True,
        )
    _print_result(record)
    return record


def _print_result(record: dict[str, Any]) -> None:
    def pct(key: str) -> str:
        if key not in record:
            return "n/a"
        return f"{record[key]:+.0f}%"

    err = record.get("err_max_pct")
    err_txt = "n/a" if err is None else f"{err:.0f}%"
    ratio = record.get("area_ratio")
    ratio_txt = "n/a" if ratio is None else f"{ratio:.2f}"
    hf = record.get("H_frozen_mm", float("nan"))
    hf_txt = "n/a" if hf != hf else f"{hf:.2f}"
    print(
        f"RESULT W={record['W_mm']:.2f} ({pct('err_W_pct')})  "
        f"D={record['D_mm']:.2f} ({pct('err_D_pct')})  "
        f"H_max={record['H_mm']:.2f} ({pct('err_H_pct')})  "
        f"H_frozen={hf_txt} ({pct('err_Hf_pct')})  "
        f"W_bead={record['W_bead_mm']:.2f}  "
        f"area={record['area_mm2']:.2f}  area_ratio={ratio_txt}  "
        f"steady={record['steady']}  err_max={err_txt}  "
        f"W_sd={record['W_sd_mm']:.2f}  frozen={record['frozen_frac']:.2f}  "
        f"tau={record['tau']:.4f}  mach={record['mach_steps']}/{record['n_steps']}  "
        f"u_max={record.get('u_pool_max_ms', 0.0):.3f}  "
        f"u_surf={record.get('u_surf_p95_ms', 0.0):.3f}  "
        f"nu_ratio={record.get('nu_ratio', 0.0):.2f}  "
        f"mach_frac={record.get('mach_frac', 0.0):.4f}  "
        f"force_clamp_frac={record.get('force_clamp_frac', 0.0):.4f}  "
        f"fl_med={record.get('surface_fl_median', 0.0):.3f}  "
        f"frac_liquid={record.get('surface_frac_liquid', 0.0):.2f}  "
        f"atten={record.get('surface_atten_median', 0.0):.2f}  "
        f"u_ceiling={record.get('u_ceiling_m_s', 0.0):.3f}  "
        f"dx={record['dx_mm']:.3f}  "
        f"wall={record['wall_s']:.0f}s ({record['ms_per_step']:.1f} ms/step)",
        flush=True,
    )
    if not record["steady"]:
        print(f"WARNING not steady: frozen_frac={record['frozen_frac']:.3f}", flush=True)


def _summarise(records: list[dict[str, Any]]) -> None:
    if len(records) < 2:
        return
    import numpy as np

    for key in ("W_mm", "D_mm", "H_mm", "H_frozen_mm", "area_mm2", "mach_steps"):
        vals = np.array([r[key] for r in records], dtype=float)
        print(
            f"SPREAD {key}: mean={vals.mean():.3f}  sigma={vals.std(ddof=1):.3f}  "
            f"min={vals.min():.3f}  max={vals.max():.3f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--steps", type=int, default=62500)
    ap.add_argument("--time-s", type=float, default=None,
                    help="Physical time. Overrides --steps using the built dt.")
    ap.add_argument("--x0", type=float, default=22.0)
    ap.add_argument("--x1", type=float, default=26.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--a-front", type=float, default=None)
    ap.add_argument("--a-rear", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--marangoni", type=float, default=None)
    ap.add_argument("--contact", type=float, default=None)
    ap.add_argument("--sulphur", type=float, default=None)
    ap.add_argument("--arc-sigma", type=float, default=None,
                    help="Arc sigma in mm. Deposition footprint and Lorentz spread, not Goldak heat.")
    ap.add_argument("--dx", type=float, default=None, help="dx in mm")
    ap.add_argument("--lorentz-every-s", type=float, default=2.5e-4)
    ap.add_argument("--cs", type=float, default=None, help="Smagorinsky Cs override")
    ap.add_argument("--force-limit-m-s2", type=float, default=None,
                    help="Physical body-force cap. 0 switches it off so --force-limit-lu applies.")
    ap.add_argument("--force-limit-lu", type=float, default=None,
                    help="Lattice body-force cap, used when the physical cap is 0.")
    ap.add_argument("--c-darcy", type=float, default=None,
                    help="Mushy-zone C_darcy. 0 turns the brake off. Still scales with dt/dt_ref.")
    ap.add_argument("--dt-scale", type=float, default=None,
                    help="Time-step scale written into the job before the grid is built.")
    ap.add_argument("--sample-every-s", type=float, default=0.1,
                    help="Pool-flow sample interval during the last 1 s")
    ap.add_argument("--hash-fields", action="store_true",
                    help="Print sha256 of phi, flags, and H after the weld (determinism gate).")
    ap.add_argument("--no-lorentz", action="store_true",
                    help="Disable Lorentz on the twin (does not edit the job).")
    args = ap.parse_args(argv)

    job_path = pathlib.Path(args.job)
    if args.time_s is not None:
        limit_s = _path_limit_s(load_job_config(args.job))
        if float(args.time_s) > limit_s + 1e-6:
            raise SystemExit(
                f"--time-s {args.time_s} passes the path end at {limit_s:.3f} s; "
                f"a parked torch would keep heating the window"
            )
    if args.dx is not None or args.sulphur is not None or args.dt_scale is not None:
        job_path = _write_override_job(
            job_path, dx_mm=args.dx, sulphur_ppm=args.sulphur, dt_scale=args.dt_scale,
        )

    out_path = pathlib.Path(args.out) if args.out else None
    if out_path is not None and not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    records: list[dict[str, Any]] = []
    for i in range(int(args.repeat)):
        if i:
            reset_taichi()
        print(f"RUN {i + 1}/{args.repeat}", flush=True)
        record = run_once(args, job_path)
        record["repeat_index"] = i
        records.append(record)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
    _summarise(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
