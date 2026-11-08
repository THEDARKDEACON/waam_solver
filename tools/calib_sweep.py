"""Parameter sweep → Gaussian-process surrogate → calibrated parameter ranges.

For calibration with little data: instead of one best-fit point, find the
range of parameters consistent with the measurements, see which parameters
the data actually constrains, and check the result at held-out process
points.

    # 1. design (Latin hypercube over the parameter ranges) and time estimate
    python -m waam_twin.tools.calib_sweep plan  waam_twin/jobs/calibration/sweep_park.yaml
    # 2. run every sample at every calibration point (resumable; one process per run)
    python -m waam_twin.tools.calib_sweep run   waam_twin/jobs/calibration/sweep_park.yaml
    # 3. surrogate accuracy, sensitivity, posterior ranges
    python -m waam_twin.tools.calib_sweep fit   waam_twin/jobs/calibration/sweep_park.yaml
    # 4. run the best fit + posterior draws at the held-out points
    python -m waam_twin.tools.calib_sweep validate waam_twin/jobs/calibration/sweep_park.yaml

Each simulation is ``tools/steady_bead`` (steady cross-section W/D/H behind
the torch), so one sample costs one steady bead per process point.

Statistics (deliberately simple, documented in the spec file):
  - Surrogate: one GP per output per point, RBF (ARD) + white noise, inputs
    scaled to [0, 1]. Accuracy is reported as leave-one-out error.
  - Sensitivity: first-order and total Sobol indices of the surrogate
    (Saltelli / Jansen estimators).
  - Posterior: prior (uniform or truncated normal) × Gaussian likelihood with
    variance = measurement² + surrogate² + (discrepancy × measured)², over
    dense Monte-Carlo samples of the box (importance weights). With ≤ 5
    parameters this is accurate and needs no MCMC tuning.
  - A parameter is reported "constrained" when its 90 % posterior interval is
    under half its prior width; otherwise the data do not pin it down and it
    should be fixed from physics, not fitted.
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
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import yaml

from waam_twin.paths import PROJECT_ROOT

OUTPUTS = ("W_mm", "D_mm", "H_frozen_mm")
# Job ``reference:`` keys for each output.
REF_KEYS = {"W_mm": "pool_width_mm", "D_mm": "pool_depth_mm", "H_frozen_mm": "bead_height_mm"}
# Parameters steady_bead can override, and its flag for each.
PARAM_FLAGS = {
    "eta": "--eta",
    "a_front": "--a-front",
    "a_rear": "--a-rear",
    "b": "--b",
    "c": "--c",
    "marangoni": "--marangoni",
    "contact": "--contact",
    "sulphur": "--sulphur",
    "arc_sigma": "--arc-sigma",
}


# --------------------------------------------------------------------- spec
@dataclass
class Param:
    name: str
    lo: float
    hi: float
    mean: float | None = None  # truncated-normal prior when given
    sd: float | None = None

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.sd is None:
            return np.zeros_like(x)
        return -0.5 * ((x - self.mean) / self.sd) ** 2


@dataclass
class Point:
    name: str
    job: str
    reference: dict[str, tuple[float, float]] = field(default_factory=dict)  # output -> (value, sigma)


@dataclass
class Spec:
    path: pathlib.Path
    params: list[Param]
    points: list[Point]
    heldout: list[Point]
    n: int
    seed: int
    time_s: float
    x0: float
    x1: float
    dx: float | None
    fixed: dict[str, float]
    discrepancy: float
    out: pathlib.Path
    extra_args: list[str]


def _resolve(p: str) -> pathlib.Path:
    q = pathlib.Path(p)
    if q.is_absolute():
        return q
    for base in (pathlib.Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent):
        if (base / q).exists():
            return (base / q).resolve()
    return (PROJECT_ROOT / q).resolve()


def _point(d: dict, default_sigma_mm: float, default_sigma_rel: float) -> Point:
    job = _resolve(d["job"])
    jobdoc = yaml.safe_load(job.read_text(encoding="utf-8")) or {}
    jref = jobdoc.get("reference") or {}
    ref: dict[str, tuple[float, float]] = {}
    override = d.get("reference") or {}
    for out in OUTPUTS:
        if out in override:
            v = override[out]
            val, sig = (v if isinstance(v, (list, tuple)) else (v, None))
        elif not jref.get("awaiting_measurement") and jref.get(REF_KEYS[out]) is not None:
            val, sig = jref[REF_KEYS[out]], None
        else:
            continue
        if val is None:
            continue
        val = float(val)
        if sig is None:
            sig = max(default_sigma_mm, default_sigma_rel * abs(val))
        ref[out] = (val, float(sig))
    return Point(name=str(d.get("name") or job.stem), job=str(job), reference=ref)


def load_spec(path: str | pathlib.Path) -> Spec:
    path = _resolve(str(path))
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meas = d.get("measurement_sigma") or {}
    s_mm = float(meas.get("abs_mm", 0.15))
    s_rel = float(meas.get("rel", 0.05))
    params = []
    for name, v in (d.get("params") or {}).items():
        if name not in PARAM_FLAGS:
            raise ValueError(f"unknown parameter {name!r}; steady_bead overrides: {sorted(PARAM_FLAGS)}")
        if isinstance(v, dict):
            params.append(Param(name, float(v["lo"]), float(v["hi"]),
                                v.get("mean"), v.get("sd")))
        else:
            params.append(Param(name, float(v[0]), float(v[1])))
    if not params:
        raise ValueError("spec needs at least one entry under params:")
    points = [_point(p, s_mm, s_rel) for p in d.get("points") or []]
    if not points:
        raise ValueError("spec needs at least one calibration point under points:")
    heldout = [_point(p, s_mm, s_rel) for p in d.get("heldout") or []]
    fixed = {k: float(v) for k, v in (d.get("fixed") or {}).items()}
    for k in fixed:
        if k not in PARAM_FLAGS:
            raise ValueError(f"unknown fixed parameter {k!r}")
    out = d.get("out") or f"runs/calibration/{path.stem}.jsonl"
    out_p = pathlib.Path(out)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    return Spec(
        path=path, params=params, points=points, heldout=heldout,
        n=int(d.get("n", 8 * len(params))), seed=int(d.get("seed", 1)),
        time_s=float(d.get("time_s", 2.5)), x0=float(d.get("x0", 22.0)), x1=float(d.get("x1", 26.0)),
        dx=None if d.get("dx_mm") is None else float(d["dx_mm"]),
        fixed=fixed, discrepancy=float(d.get("model_discrepancy_rel", 0.05)),
        out=out_p, extra_args=[str(a) for a in d.get("steady_bead_args") or []],
    )


# ------------------------------------------------------------------- design
def lhs_design(spec: Spec) -> np.ndarray:
    """n × d samples in parameter units (maximin-optimised Latin hypercube)."""
    from scipy.stats import qmc

    d = len(spec.params)
    sampler = qmc.LatinHypercube(d=d, seed=spec.seed, optimization="random-cd")
    u = sampler.random(spec.n)
    lo = np.array([p.lo for p in spec.params])
    hi = np.array([p.hi for p in spec.params])
    return lo + u * (hi - lo)


def _unit(spec: Spec, X: np.ndarray) -> np.ndarray:
    lo = np.array([p.lo for p in spec.params])
    hi = np.array([p.hi for p in spec.params])
    return (np.asarray(X, dtype=np.float64) - lo) / (hi - lo)


# ---------------------------------------------------------------------- run
Runner = Callable[[Point, dict[str, float], Spec], dict[str, Any]]


def steady_bead_runner(point: Point, values: dict[str, float], spec: Spec) -> dict[str, Any]:
    """One steady bead in its own process (frees the GPU between runs)."""
    tmp = spec.out.parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rec_path = tmp / f"rec_{os.getpid()}_{time.time_ns()}.jsonl"
    cmd = [sys.executable, "-m", "waam_twin.tools.steady_bead", "--job", point.job,
           "--time-s", str(spec.time_s), "--x0", str(spec.x0), "--x1", str(spec.x1),
           "--out", str(rec_path)]
    if spec.dx is not None:
        cmd += ["--dx", str(spec.dx)]
    for k, v in {**spec.fixed, **values}.items():
        cmd += [PARAM_FLAGS[k], repr(float(v))]
    cmd += spec.extra_args
    log = spec.out.with_suffix(".log")
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {point.name} {values}\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
    if proc.returncode != 0 or not rec_path.exists():
        return {"error": f"steady_bead exit {proc.returncode} (see {log})"}
    rec = json.loads(rec_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    rec_path.unlink(missing_ok=True)
    return rec


def _load(out: pathlib.Path) -> list[dict]:
    if not out.exists():
        return []
    rows = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_design(spec: Spec, runner: Runner = steady_bead_runner, kind: str = "design",
               cases: list[tuple[int, dict[str, float]]] | None = None,
               points: list[Point] | None = None) -> list[dict]:
    """Run every (sample, point) not yet in spec.out; append one row each."""
    spec.out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["kind"], r["sample"], r["point"]) for r in _load(spec.out) if "error" not in r}
    if cases is None:
        X = lhs_design(spec)
        cases = [(i, {p.name: float(X[i, j]) for j, p in enumerate(spec.params)}) for i in range(len(X))]
    points = spec.points if points is None else points
    todo = [(i, v, pt) for i, v in cases for pt in points if (kind, i, pt.name) not in done]
    print(f"[calib_sweep] {len(todo)} runs to do ({len(done)} already in {spec.out})", flush=True)
    rows = []
    for n, (i, values, pt) in enumerate(todo, 1):
        t0 = time.time()
        rec = runner(pt, values, spec)
        row = {"kind": kind, "sample": i, "point": pt.name, "params": values,
               **{k: rec.get(k) for k in OUTPUTS},
               "mach_frac": rec.get("mach_frac"), "force_clamp_frac": rec.get("force_clamp_frac"),
               "wall_s": time.time() - t0}
        if "error" in rec:
            row["error"] = rec["error"]
        with spec.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        outs = "  ".join(f"{k}={row[k]:.2f}" for k in OUTPUTS if isinstance(row.get(k), (int, float)))
        print(f"[calib_sweep] {n}/{len(todo)} {kind} #{i} @ {pt.name}: {outs or row.get('error')}  "
              f"({row['wall_s']:.0f} s)", flush=True)
    return rows


# ---------------------------------------------------------------------- fit
def _gp(X: np.ndarray, y: np.ndarray):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(length_scale=np.full(d, 0.5), length_scale_bounds=(0.05, 20.0))
              + WhiteKernel(1e-3, (1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=4,
                                  random_state=0)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        # A length scale at its upper bound just means "this parameter barely
        # matters" (the Sobol indices say the same); not a fit failure.
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


@dataclass
class Surrogate:
    """GP per (point, output), trained on unit-scaled parameters."""
    models: dict[tuple[str, str], Any]
    loo: dict[tuple[str, str], dict[str, float]]

    def predict(self, point: str, out: str, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.models[(point, out)].predict(U, return_std=True)
        return mu, sd


def fit_surrogates(spec: Spec, rows: list[dict], loo: bool = True) -> Surrogate:
    models, loo_err = {}, {}
    for pt in spec.points:
        prow = [r for r in rows if r["kind"] == "design" and r["point"] == pt.name and "error" not in r]
        for out in OUTPUTS:
            good = [r for r in prow if isinstance(r.get(out), (int, float)) and math.isfinite(r[out])]
            if len(good) < len(spec.params) + 2:
                continue
            X = _unit(spec, [[r["params"][p.name] for p in spec.params] for r in good])
            y = np.array([r[out] for r in good], dtype=np.float64)
            models[(pt.name, out)] = _gp(X, y)
            if loo:
                errs = []
                for i in range(len(y)):
                    m = np.ones(len(y), bool)
                    m[i] = False
                    g = _gp(X[m], y[m])
                    errs.append(float(g.predict(X[i:i + 1])[0] - y[i]))
                e = np.array(errs)
                loo_err[(pt.name, out)] = {
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "max_abs": float(np.max(np.abs(e))),
                    "rel_rmse_pct": float(100 * np.sqrt(np.mean(e ** 2)) / max(np.mean(np.abs(y)), 1e-9)),
                    "spread": float(np.std(y)),
                }
    return Surrogate(models, loo_err)


def sobol_indices(f: Callable[[np.ndarray], np.ndarray], d: int, n: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """First-order (Saltelli 2010) and total (Jansen) indices on the unit cube."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, d))
    B = rng.random((n, d))
    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    S1, ST = np.zeros(d), np.zeros(d)
    if var <= 0:
        return S1, ST
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = f(ABi)
        S1[i] = np.mean(fB * (fABi - fA)) / var
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var
    return S1, ST


def posterior(spec: Spec, sur: Surrogate, n_mc: int = 200_000, seed: int = 0) -> dict[str, Any]:
    """Importance-weighted Monte-Carlo posterior over the parameter box."""
    rng = np.random.default_rng(seed)
    d = len(spec.params)
    U = rng.random((n_mc, d))
    lo = np.array([p.lo for p in spec.params])
    hi = np.array([p.hi for p in spec.params])
    X = lo + U * (hi - lo)
    logw = np.zeros(n_mc)
    for j, p in enumerate(spec.params):
        logw += p.log_prior(X[:, j])
    used = []
    for pt in spec.points:
        for out, (val, sig) in pt.reference.items():
            if (pt.name, out) not in sur.models:
                continue
            mu, sd = sur.predict(pt.name, out, U)
            var = sig ** 2 + sd ** 2 + (spec.discrepancy * val) ** 2
            logw += -0.5 * (mu - val) ** 2 / var - 0.5 * np.log(var)
            used.append(f"{pt.name}:{out}")
    if not used:
        raise ValueError("no measured outputs to calibrate against (check reference: blocks)")
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    summary = {}
    for j, p in enumerate(spec.params):
        x = X[:, j]
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        q = lambda a: float(x[order][min(np.searchsorted(cw, a), len(cw) - 1)])  # noqa: E731
        q05, q50, q95 = q(0.05), q(0.5), q(0.95)
        width = (q95 - q05) / (p.hi - p.lo)
        summary[p.name] = {
            "mean": float(np.sum(w * x)), "median": q50, "q05": q05, "q95": q95,
            "prior": [p.lo, p.hi], "interval_frac_of_prior": float(width),
            "constrained": bool(width < 0.5),
        }
    best = int(np.argmax(w))
    draws_idx = rng.choice(n_mc, size=min(2000, n_mc), p=w)
    return {
        "used_outputs": used,
        "ess": ess,
        "params": summary,
        "map": {p.name: float(X[best, j]) for j, p in enumerate(spec.params)},
        "draws": X[draws_idx].tolist(),
        "fit_at_map": {
            f"{pt.name}:{out}": {
                "measured": val, "sigma": sig,
                "predicted": float(sur.predict(pt.name, out, U[best:best + 1])[0][0]),
            }
            for pt in spec.points for out, (val, sig) in pt.reference.items()
            if (pt.name, out) in sur.models
        },
    }


def fit_report(spec: Spec, rows: list[dict] | None = None, loo: bool = True) -> dict[str, Any]:
    rows = _load(spec.out) if rows is None else rows
    design = [r for r in rows if r["kind"] == "design"]
    bad = [r for r in design if "error" in r]
    clamped = [r for r in design if (r.get("mach_frac") or 0) > 0.01 or (r.get("force_clamp_frac") or 0) > 0.01]
    sur = fit_surrogates(spec, rows, loo=loo)
    d = len(spec.params)
    sens = {}
    for (pt, out), gp in sur.models.items():
        S1, ST = sobol_indices(lambda U, g=gp: g.predict(U), d)
        sens[f"{pt}:{out}"] = {p.name: {"S1": float(S1[j]), "ST": float(ST[j])} for j, p in enumerate(spec.params)}
    post = posterior(spec, sur)
    report = {
        "spec": str(spec.path), "n_design_runs": len(design), "n_failed": len(bad),
        "n_clamp_limited": len(clamped),
        "surrogate_loo": {f"{k[0]}:{k[1]}": v for k, v in sur.loo.items()},
        "sensitivity": sens, "posterior": {k: v for k, v in post.items() if k != "draws"},
    }
    rep_path = spec.out.with_suffix(".report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    spec.out.with_suffix(".draws.json").write_text(json.dumps({
        "params": [p.name for p in spec.params], "map": post["map"], "draws": post["draws"]}), encoding="utf-8")
    _print_report(spec, report)
    return report


def _print_report(spec: Spec, rep: dict) -> None:
    print(f"\n== calib_sweep: {spec.path.name}  ({rep['n_design_runs']} design runs, "
          f"{rep['n_failed']} failed, {rep['n_clamp_limited']} clamp-limited >1 % of steps)")
    if rep["n_clamp_limited"]:
        print("   NOTE: clamp-limited runs report the numerical speed cap, not physics.")
    print("\n-- surrogate leave-one-out error (should be well below the measurement sigma)")
    for k, v in rep["surrogate_loo"].items():
        print(f"   {k:28s} rmse {v['rmse']:.3f} mm ({v['rel_rmse_pct']:.1f} %)  max {v['max_abs']:.3f}  "
              f"output spread {v['spread']:.3f}")
    print("\n-- sensitivity (Sobol total index ST; S1 = alone)")
    names = [p.name for p in spec.params]
    print("   " + " " * 28 + "".join(f"{n:>12s}" for n in names))
    for k, v in rep["sensitivity"].items():
        print(f"   {k:28s}" + "".join(f"{v[n]['ST']:>7.2f}/{v[n]['S1']:.2f}" for n in names))
    post = rep["posterior"]
    print(f"\n-- posterior from {', '.join(post['used_outputs'])}  (effective samples {post['ess']:.0f})")
    for n, s in post["params"].items():
        tag = "constrained" if s["constrained"] else "NOT constrained → fix from physics"
        print(f"   {n:10s} median {s['median']:.4g}  90% [{s['q05']:.4g}, {s['q95']:.4g}]  "
              f"prior [{s['prior'][0]:.4g}, {s['prior'][1]:.4g}]  ({s['interval_frac_of_prior'] * 100:.0f} % of prior) {tag}")
    print("   best fit:", {k: round(v, 4) for k, v in post["map"].items()})
    for k, v in post["fit_at_map"].items():
        print(f"   {k:28s} measured {v['measured']:.2f} ± {v['sigma']:.2f}   surrogate at best fit {v['predicted']:.2f}")


# ----------------------------------------------------------------- validate
def validate(spec: Spec, n_draws: int = 4, runner: Runner = steady_bead_runner) -> dict[str, Any]:
    dpath = spec.out.with_suffix(".draws.json")
    if not dpath.exists():
        raise SystemExit("run `fit` first (no posterior draws file)")
    if not spec.heldout:
        raise SystemExit("spec has no heldout: points")
    dd = json.loads(dpath.read_text(encoding="utf-8"))
    names = dd["params"]
    rng = np.random.default_rng(spec.seed)
    draws = [dd["map"]] + [dict(zip(names, map(float, dd["draws"][i])))
                           for i in rng.choice(len(dd["draws"]), size=min(n_draws, len(dd["draws"])), replace=False)]
    cases = list(enumerate(draws))
    run_design(spec, runner=runner, kind="validate", cases=cases, points=spec.heldout)
    rows = [r for r in _load(spec.out) if r["kind"] == "validate" and "error" not in r]
    result = {}
    print("\n-- held-out prediction (best fit = sample 0; band = posterior draws)")
    for pt in spec.heldout:
        prow = [r for r in rows if r["point"] == pt.name]
        for out in OUTPUTS:
            vals = [r[out] for r in prow if isinstance(r.get(out), (int, float))]
            if not vals:
                continue
            best = next((r[out] for r in prow if r["sample"] == 0), vals[0])
            entry = {"best": best, "min": min(vals), "max": max(vals)}
            line = f"   {pt.name:28s} {out:12s} best {best:.2f}  band [{min(vals):.2f}, {max(vals):.2f}]"
            if out in pt.reference:
                val, sig = pt.reference[out]
                err = 100.0 * (best - val) / val
                entry.update(measured=val, sigma=sig, err_pct=err)
                line += f"  measured {val:.2f} ± {sig:.2f}  error {err:+.1f} %"
            else:
                line += "  (no measurement: prediction only)"
            print(line)
            result[f"{pt.name}:{out}"] = entry
    spec.out.with_suffix(".validate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------- CLI
def _plan(spec: Spec) -> None:
    X = lhs_design(spec)
    print(f"[calib_sweep] {spec.path.name}: {len(spec.params)} parameters, {spec.n} samples × "
          f"{len(spec.points)} calibration point(s) = {spec.n * len(spec.points)} steady beads "
          f"(+ held-out validation later)")
    for p in spec.params:
        prior = f"normal({p.mean}, {p.sd}) truncated" if p.mean is not None else "uniform"
        print(f"   {p.name:10s} [{p.lo}, {p.hi}]  prior {prior}")
    for pt in spec.points + spec.heldout:
        role = "calibrate" if pt in spec.points else "held-out "
        refs = ", ".join(f"{k}={v[0]}±{v[1]:.2f}" for k, v in pt.reference.items()) or "no measurement"
        print(f"   {role} {pt.name}: {refs}")
    if len(spec.points) * sum(len(p.reference) for p in spec.points) < len(spec.params):
        print("   WARNING: fewer measured outputs than parameters — expect unconstrained parameters.")
    print(f"   first samples:\n" + "\n".join(
        "     " + ", ".join(f"{p.name}={X[i, j]:.4g}" for j, p in enumerate(spec.params)) for i in range(min(4, len(X)))))
    print(f"   output → {spec.out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "run", "fit", "validate", "report"))
    ap.add_argument("spec")
    ap.add_argument("--no-loo", action="store_true", help="skip leave-one-out surrogate check (faster fit)")
    ap.add_argument("--draws", type=int, default=4, help="posterior draws to run at held-out points")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.command == "plan":
        _plan(spec)
    elif a.command == "run":
        run_design(spec)
    elif a.command in ("fit", "report"):
        fit_report(spec, loo=not a.no_loo)
    elif a.command == "validate":
        validate(spec, n_draws=a.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
