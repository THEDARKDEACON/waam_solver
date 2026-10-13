"""
energy_ledger.py — run a short bead and print energy/mass ledger closure.

Usage:
  PYTHONPATH=.. WAAM_BACKEND=cuda python -m waam_twin.tools.diagnostics.energy_ledger \\
      --job jobs/examples/bead_calibrate.yaml --steps 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", default="jobs/examples/bead_calibrate.yaml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--backend", default=os.environ.get("WAAM_BACKEND", "cuda"))
    p.add_argument("--tol", type=float, default=0.02, help="max relative closure residual")
    args = p.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    from waam_twin.runtime import init_taichi
    from waam_twin import WAAMTwin
    from waam_twin.job import load_job_config, parse_torch_path
    from waam_twin.torch_path import TorchPathDriver
    from waam_twin.physics.ledger import ensure_ledger

    init_taichi(backend=args.backend)
    twin = WAAMTwin.from_job(args.job)
    twin.reset()
    led = ensure_ledger(twin)
    # Seed stored-0 after reset
    snap0 = led.snapshot(twin)

    job = load_job_config(args.job)
    drv = TorchPathDriver(parse_torch_path(job), twin.travel_speed_m_s)
    it = drv.positions_for_steps(args.steps, twin.grid.dt)
    for _ in range(args.steps):
        _, x, y, z = next(it)
        twin.step(x, y, True, torch_z_m=z)

    snap = led.snapshot(twin)
    out = snap.to_dict()
    out["E_stored_0_at_reset_J"] = snap0.E_stored_J
    print(json.dumps(out, indent=2))
    ok = snap.energy_closure_rel() <= args.tol
    # Cap energy on step 1 is a known H6 defect — flag it
    if snap.E_cap_J > 0.1 * max(snap.E_stored_0_J, 1.0):
        print(
            f"[WARN] large enthalpy-cap cumulative {snap.E_cap_J:.1f} J "
            f"(likely H6 uninitialised cp field)",
            file=sys.stderr,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
