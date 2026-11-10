"""Mesh-convergence study for the coupled bead (frozen physical inputs).

Runs ``tools/steady_bead`` once per ``dx`` (each in its own process, so GPU
memory is released between meshes), then reports per-mesh diagnostics and a
Roache grid-convergence index (GCI, Celik et al. 2008 procedure for
non-constant refinement ratios) for W, D, H_frozen and bead area.

Everything that is held fixed is physical: dx changes the grid only. dt
follows the job (``auto_dt_ma``), C_darcy is rescaled by dt/dt_ref (constant
SI damping), Goldak axes and footprints are in mm, and the slice window
(--x0/--x1) is in mm.

Per-mesh flags (why a mesh may not be trustworthy):
  CLAMP   Mach or force clamp fired on > 1 % of steps (flow capped by numerics)
  LES     Smagorinsky eddy viscosity exceeds the molecular one (nu_t/nu > 1):
          the resolved flow depends on Cs·dx, so W/D drift with dx by design
  TAU     effective relaxation time 0.5 + 3 ν(1 + nu_t/nu) dt/dx² < 0.505
          (marginal stability for TRT at this mesh)
  UNSTEADY  the slice window was not in steady state

Usage (repo root on PYTHONPATH, GPU backend):

  # plan only: grid, cells, GPU memory per mesh
  python -m waam_twin.tools.mesh_study --dry-run

  # full study, coarse → fine; resumable
  WAAM_BACKEND=cuda python -m waam_twin.tools.mesh_study \\
      --dx 0.5,0.4,0.3,0.25,0.2 --time-s 2.5 --out runs/mesh_study/park.jsonl

  # report from an existing results file
  python -m waam_twin.tools.mesh_study --report runs/mesh_study/park.jsonl

Extra steady_bead options go after ``--`` (e.g. ``-- --cs 0.12``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

from waam_twin.paths import PROJECT_ROOT

DEFAULT_JOB = "jobs/examples/bead_calibrate.yaml"
QUANTITIES = ("W_mm", "D_mm", "H_frozen_mm", "area_mm2")

# Measured on an RTX 3050 4 GB, FSLBM + Lorentz + full tier, Nov 2026:
# 528 MiB @ 1.24 M cells, 1072 @ 2.94 M, 1776 @ 5.12 M, 3344 @ 10.0 M.
VRAM_BASE_MIB = 130.0
VRAM_MIB_PER_MCELL = 320.0
VRAM_MARGIN = 1.10

CLAMP_FRAC = 0.01
TAU_MARGINAL = 0.505


# ── GCI (Celik et al. 2008, J. Fluids Eng. 130:078001) ──────────────────────

def observed_order(
    h: tuple[float, float, float], phi: tuple[float, float, float],
) -> dict[str, Any]:
    """Three-grid observed order, extrapolated value and fine-grid GCI.

    ``h`` and ``phi`` are ordered fine → coarse (h1 < h2 < h3).
    Returns keys: p, phi_ext, gci_fine_pct, e21_pct, convergence
    ("monotone", "oscillatory" or "converged").
    """
    h1, h2, h3 = h
    f1, f2, f3 = phi
    r21 = h2 / h1
    r32 = h3 / h2
    e21 = f2 - f1
    e32 = f3 - f2
    scale = max(abs(f1), 1e-12)
    out: dict[str, Any] = {
        "r21": r21, "r32": r32, "e21_pct": 100.0 * abs(e21) / scale,
    }
    if abs(e21) < 1e-9 * scale and abs(e32) < 1e-9 * scale:
        out.update(p=float("nan"), phi_ext=f1, gci_fine_pct=0.0, convergence="converged")
        return out
    if abs(e21) < 1e-12 or e32 / e21 < 0.0:
        # Oscillatory (or e21 = 0): no observed order. Use the spread as a
        # conservative band: GCI ≈ 1.25 · max |φ_i − φ_1| / |φ_1|.
        spread = max(abs(e21), abs(f3 - f1))
        out.update(
            p=float("nan"), phi_ext=f1,
            gci_fine_pct=125.0 * spread / scale, convergence="oscillatory",
        )
        return out
    s = 1.0
    # Fixed-point iteration from p = 1 (the closed-form start is 0 when
    # e32 = e21, which makes r^p − 1 vanish).
    p = 1.0
    for _ in range(200):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = max(abs(math.log(abs(e32 / e21)) + q) / math.log(r21), 1e-3)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    rp = r21 ** p
    phi_ext = (rp * f1 - f2) / (rp - 1.0)
    gci = 1.25 * abs(e21 / f1) / (rp - 1.0) * 100.0
    out.update(p=p, phi_ext=phi_ext, gci_fine_pct=gci, convergence="monotone")
    return out


# ── planning ────────────────────────────────────────────────────────────────

def _detect_vram_mib() -> float | None:
    try:
        txt = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        total, used = (float(v) for v in txt.split(","))
        return total - used
    except Exception:
        return None


def vram_need_mib(n_cells: int) -> float:
    return (VRAM_BASE_MIB + VRAM_MIB_PER_MCELL * n_cells / 1e6) * VRAM_MARGIN


def plan(job_path: pathlib.Path, dx_list: list[float], vram_mib: float | None) -> list[dict]:
    from waam_twin.job import load_job_config
    from waam_twin.runtime import grid_at_dx

    job = load_job_config(job_path)
    domain = tuple(float(v) for v in job["simulation"]["domain_mm"])
    base = None
    rows = []
    for dx in dx_list:
        nx, ny, nz, _ = grid_at_dx(domain, dx)
        n = nx * ny * nz
        # Work ∝ cells × steps, steps ∝ 1/dt ∝ 1/dx (acoustic scaling).
        work = n / dx
        base = base or work
        need = vram_need_mib(n)
        rows.append({
            "dx_mm": dx, "grid": f"{nx}×{ny}×{nz}", "n_cells": n,
            "vram_need_mib": need,
            "fits": (vram_mib is None) or (need <= vram_mib),
            "rel_cost": work / base,
        })
    return rows


def _print_plan(rows: list[dict], vram_mib: float | None) -> None:
    avail = "unknown" if vram_mib is None else f"{vram_mib:.0f} MiB free"
    print(f"PLAN  (GPU: {avail}; estimate incl. {int((VRAM_MARGIN - 1) * 100)}% margin)")
    for r in rows:
        print(
            f"  dx={r['dx_mm']:<5g} grid={r['grid']:<14} cells={r['n_cells'] / 1e6:5.2f}M  "
            f"GPU≈{r['vram_need_mib']:5.0f} MiB  {'fits' if r['fits'] else 'DOES NOT FIT'}  "
            f"cost ×{r['rel_cost']:.1f} vs dx={rows[0]['dx_mm']:g}"
        )


# ── running ─────────────────────────────────────────────────────────────────

def _done_dx(out: pathlib.Path) -> set[float]:
    if not out.exists():
        return set()
    done = set()
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(round(float(json.loads(line)["dx_mm"]), 4))
    return done


def run_mesh(
    job: pathlib.Path, dx: float, args: argparse.Namespace,
    out: pathlib.Path, extra: list[str],
) -> int:
    log_path = out.with_suffix(f".dx{dx:g}.log")
    cmd = [
        sys.executable, "-m", "waam_twin.tools.steady_bead",
        "--job", str(job), "--dx", f"{dx:g}",
        "--time-s", f"{args.time_s:g}", "--x0", f"{args.x0:g}", "--x1", f"{args.x1:g}",
        "--out", str(out), *extra,
    ]
    print(f"RUN dx={dx:g}  log → {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    print(f"  dx={dx:g} exit={proc.returncode}  {time.time() - t0:.0f}s", flush=True)
    return proc.returncode


# ── report ──────────────────────────────────────────────────────────────────

def _load(out: pathlib.Path) -> dict[float, dict]:
    """Records grouped by dx; repeats are averaged."""
    groups: dict[float, list[dict]] = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            groups.setdefault(round(float(r["dx_mm"]), 4), []).append(r)
    merged = {}
    for dx, recs in groups.items():
        m = dict(recs[0])
        for key in (*QUANTITIES, "nu_ratio", "mach_frac", "force_clamp_frac", "tau", "dt_s"):
            vals = [float(r[key]) for r in recs if r.get(key) is not None
                    and not (isinstance(r[key], float) and math.isnan(r[key]))]
            m[key] = sum(vals) / len(vals) if vals else float("nan")
        for key in ("mach_frac_window", "force_clamp_frac_window"):
            # Only where recorded: older files have no window counts, and a NaN
            # here would silently clear the CLAMP flag.
            vals = [float(r[key]) for r in recs if r.get(key) is not None]
            if vals:
                m[key] = sum(vals) / len(vals)
            else:
                m.pop(key, None)
        m["steady"] = all(bool(r.get("steady", True)) for r in recs)
        m["n_repeats"] = len(recs)
        merged[dx] = m
    return merged


def mesh_flags(r: dict, nu_m2_s: float | None) -> list[str]:
    flags = []
    # Clamps inside the measurement window, when recorded (a start-up burst
    # before the window does not set the steady bead); else the whole run.
    mach = r.get("mach_frac_window")
    force = r.get("force_clamp_frac_window")
    if mach is None:
        mach, force = r.get("mach_frac", 0), r.get("force_clamp_frac", 0)
    if max(float(mach or 0), float(force or 0)) > CLAMP_FRAC:
        flags.append("CLAMP")
    ratio = float(r.get("nu_ratio", 0) or 0)
    if ratio > 1.0:
        flags.append("LES")
    tau = float(r.get("tau", 0.0) or 0.0)
    if tau > 0.0:
        tau_eff = 0.5 + (tau - 0.5) * (1.0 + ratio)
        r["tau_eff"] = tau_eff
        if tau_eff < TAU_MARGINAL:
            flags.append("TAU")
    if not r.get("steady", True):
        flags.append("UNSTEADY")
    return flags


def report(out: pathlib.Path, target_dx: float, gate_pct: float) -> dict[str, Any]:
    data = _load(out)
    if not data:
        raise SystemExit(f"no records in {out}")
    dxs = sorted(data)
    print(f"\nMESH STUDY  {out}")
    print(
        f"  {'dx':>5} {'cells':>7} {'tau':>8} {'tau_eff':>8} {'nu_t/nu':>8} "
        f"{'mach%':>6} {'force%':>6} {'W':>6} {'D':>6} {'Hfz':>6} {'area':>6}  flags"
    )
    for dx in dxs:
        r = data[dx]
        r["flags"] = mesh_flags(r, None)
        print(
            f"  {dx:5.3f} {float(r.get('n_cells') or float('nan')) / 1e6:7.2f} "
            f"{r.get('tau', float('nan')):8.5f} {r.get('tau_eff', float('nan')):8.5f} "
            f"{r.get('nu_ratio', float('nan')):8.2f} "
            f"{100 * r.get('mach_frac_window', r.get('mach_frac', 0)):6.2f} "
            f"{100 * r.get('force_clamp_frac_window', r.get('force_clamp_frac', 0)):6.2f} "
            f"{r['W_mm']:6.2f} {r['D_mm']:6.2f} {r.get('H_frozen_mm', float('nan')):6.2f} "
            f"{r.get('area_mm2', float('nan')):6.2f}  {','.join(r['flags']) or 'ok'}"
        )

    summary: dict[str, Any] = {"target_dx_mm": target_dx, "gate_pct": gate_pct, "meshes": dxs}
    fine = dxs[:3]
    gate_ok = True
    n_gated = 0
    for q in QUANTITIES:
        vals = {dx: data[dx].get(q) for dx in dxs}
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals.values()):
            continue
        if len(fine) >= 3:
            res = observed_order(tuple(fine), tuple(vals[d] for d in fine))
            ref = res["phi_ext"]
            how = (
                f"p={res['p']:.2f}  φ_ext={ref:.3f}  GCI_fine={res['gci_fine_pct']:.1f}%"
                if res["convergence"] == "monotone"
                else f"{res['convergence']}  band ±{res['gci_fine_pct']:.1f}% (no order)"
            )
        else:
            res = {"convergence": "two-mesh"}
            ref = vals[dxs[0]]
            how = f"two meshes only: reference = finest ({ref:.3f})"
        err_target = None
        if round(target_dx, 4) in vals and abs(ref) > 1e-12:
            err_target = 100.0 * abs(vals[round(target_dx, 4)] - ref) / abs(ref)
        summary[q] = {**res, "reference": ref, "err_at_target_pct": err_target}
        tgt = "n/a" if err_target is None else f"{err_target:.1f}%"
        if res["convergence"] == "two-mesh" and round(target_dx, 4) == dxs[0]:
            err_target = None
            summary[q]["err_at_target_pct"] = None
            tgt = "n/a (target is the reference mesh; add a finer dx)"
        print(f"  {q:<12} {how}   error at dx={target_dx:g}: {tgt}")
        if q in ("W_mm", "D_mm") and err_target is not None:
            n_gated += 1
            if err_target > gate_pct:
                gate_ok = False
    flagged = {dx: data[dx]["flags"] for dx in dxs if data[dx]["flags"]}
    summary["flags"] = flagged
    verdict = ("PASS" if gate_ok else "FAIL") if n_gated == 2 else "INCONCLUSIVE"
    summary["gate_pass"] = gate_ok if n_gated == 2 else None
    summary["gate"] = verdict
    print(f"  GATE (W, D within {gate_pct:g}% at dx={target_dx:g}): {verdict}")
    if flagged:
        print(
            "  NOTE flagged meshes: CLAMP/LES/TAU mean the flow there is set by "
            "numerics (clamps, Cs·dx, near-0.5 τ), so their W/D are not a "
            "resolution sequence; prefer meshes marked ok for the GCI."
        )
    rep_path = out.with_suffix(".report.json")
    rep_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"  report → {rep_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default=DEFAULT_JOB)
    ap.add_argument("--dx", default="0.5,0.4,0.3,0.25,0.2", help="mm, comma-separated")
    ap.add_argument("--time-s", type=float, default=2.5)
    ap.add_argument("--x0", type=float, default=22.0)
    ap.add_argument("--x1", type=float, default=26.0)
    ap.add_argument("--out", default="runs/mesh_study/mesh_study.jsonl")
    ap.add_argument("--target-dx", type=float, default=0.4,
                    help="mesh whose error vs the extrapolated value is gated")
    ap.add_argument("--gate-pct", type=float, default=10.0)
    ap.add_argument("--vram-mib", type=float, default=None,
                    help="free GPU memory; default: nvidia-smi")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None, help="report an existing results file and exit")
    ap.add_argument("--force", action="store_true", help="rerun meshes already in --out")
    ap.add_argument("--sparse-dx", default="",
                    help="comma-separated meshes to run on the sparse grid (skips the dense "
                         "memory check; set WAAM_DEVICE_MEMORY_GB to the pool size)")
    args = ap.parse_args(argv)

    if args.report:
        p = pathlib.Path(args.report)
        report(p if p.is_absolute() else PROJECT_ROOT / p, args.target_dx, args.gate_pct)
        return 0

    job = pathlib.Path(args.job)
    job = job if job.is_absolute() else PROJECT_ROOT / job
    dx_list = sorted((float(v) for v in args.dx.split(",")), reverse=True)
    vram = args.vram_mib if args.vram_mib is not None else _detect_vram_mib()
    rows = plan(job, dx_list, vram)
    _print_plan(rows, vram)
    if args.dry_run:
        return 0

    out = pathlib.Path(args.out)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.force else _done_dx(out)
    sparse_dx = {round(float(v), 4) for v in args.sparse_dx.split(",") if v.strip()}
    for r in rows:
        dx = r["dx_mm"]
        if round(dx, 4) in done:
            print(f"SKIP dx={dx:g} (already in {out.name})")
            continue
        sparse = round(dx, 4) in sparse_dx
        if not r["fits"] and not sparse:
            print(f"SKIP dx={dx:g}: needs ≈{r['vram_need_mib']:.0f} MiB, {vram:.0f} free "
                  f"(add it to --sparse-dx to run on the sparse grid)")
            continue
        run_mesh(job, dx, args, out, extra + (["--sparse"] if sparse else []))
    if out.exists():
        report(out, args.target_dx, args.gate_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
