"""
test_calib_sweep.py — sweep → GP surrogate → posterior, on a known answer.

A synthetic "simulator" replaces steady_bead: W and D depend on eta and b,
H on eta only, and a third parameter has no effect at all. The measurements
are its output at known true parameters. The tool must

  - build a Latin-hypercube design, run it, and resume without re-running;
  - fit surrogates with small leave-one-out error;
  - put the true eta and b inside their 90 % posterior intervals and mark
    them constrained;
  - mark the no-effect parameter NOT constrained (its posterior stays near
    the prior) and give it ~zero Sobol index;
  - run held-out validation at a second process point (stronger arc there)
    and predict it within a few percent.
"""

from __future__ import annotations

import json
import shutil

import yaml

from waam_twin.paths import PROJECT_ROOT
from waam_twin.tools import calib_sweep as cs

TRUE = {"eta": 0.82, "b": 3.1, "marangoni": 1.0}
TMP = PROJECT_ROOT / ".waam_sweep_tmp" / "calib_sweep_test"


def _model(point: str, p: dict[str, float]) -> dict[str, float]:
    gain = 1.25 if point == "hot" else 1.0
    W = gain * (2.0 + 4.0 * p["eta"] + 0.6 * p["b"])
    D = gain * (0.4 + 2.2 * p["eta"] - 0.15 * p["b"])
    H = 1.0 + 1.5 * p["eta"]
    return {"W_mm": W, "D_mm": D, "H_frozen_mm": H}


def _runner(point, values, spec):
    p = {**TRUE, **spec.fixed, **values}
    return {**_model(point.name, p), "mach_frac": 0.0, "force_clamp_frac": 0.0}


def _job(name: str, ref: dict[str, float]) -> str:
    path = TMP / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"reference": ref}), encoding="utf-8")
    return str(path)


def run() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    truth = _model("flat", TRUE)
    hot = _model("hot", TRUE)
    spec_path = TMP / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({
        "points": [{"name": "flat", "job": _job("flat", {
            "pool_width_mm": truth["W_mm"], "pool_depth_mm": truth["D_mm"], "bead_height_mm": truth["H_frozen_mm"]})}],
        "heldout": [{"name": "hot", "job": _job("hot", {
            "pool_width_mm": hot["W_mm"], "pool_depth_mm": hot["D_mm"], "bead_height_mm": hot["H_frozen_mm"]})}],
        "params": {"eta": [0.70, 0.90], "b": [2.0, 4.0], "marangoni": [0.5, 1.5]},
        "n": 20, "seed": 3,
        "measurement_sigma": {"abs_mm": 0.03, "rel": 0.0},
        "model_discrepancy_rel": 0.0,
        "out": str(TMP / "sweep.jsonl"),
    }), encoding="utf-8")
    spec = cs.load_spec(spec_path)

    rows = cs.run_design(spec, runner=_runner)
    assert len(rows) == 20
    assert cs.run_design(spec, runner=_runner) == []  # resumable: nothing left to do

    rep = cs.fit_report(spec, loo=True)
    for k, v in rep["surrogate_loo"].items():
        assert v["rel_rmse_pct"] < 2.0, (k, v)
    post = rep["posterior"]["params"]
    for name in ("eta", "b"):
        s = post[name]
        assert s["q05"] <= TRUE[name] <= s["q95"], (name, s)
        assert s["constrained"], (name, s)
    assert not post["marangoni"]["constrained"], post["marangoni"]
    assert post["marangoni"]["interval_frac_of_prior"] > 0.7
    st = rep["sensitivity"]["flat:W_mm"]
    assert st["marangoni"]["ST"] < 0.02 and st["eta"]["ST"] > 0.3, st

    val = cs.validate(spec, n_draws=3, runner=_runner)
    for out in ("W_mm", "D_mm", "H_frozen_mm"):
        e = val[f"hot:{out}"]
        assert abs(e["err_pct"]) < 3.0, (out, e)
    assert json.loads((TMP / "sweep.validate.json").read_text())
    print(f"[calib_sweep] eta {post['eta']['median']:.3f} [{post['eta']['q05']:.3f}, {post['eta']['q95']:.3f}]  "
          f"b {post['b']['median']:.2f} [{post['b']['q05']:.2f}, {post['b']['q95']:.2f}]  "
          f"marangoni unconstrained ({post['marangoni']['interval_frac_of_prior'] * 100:.0f} % of prior)")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    run()
    print("PASS")
