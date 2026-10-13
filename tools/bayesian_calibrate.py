"""
Phase 7 — global calibration of arc-radius law (+ Tier-C λ only).

Fits σ0, a_I, a_V, a_ctwd (and optionally enhanced_k_factor for Tier C) via a
Gaussian-process surrogate + simple MCMC. Physical material / force parameters
are frozen from independent sources.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass, asdict
from typing import Any, Callable


@dataclass
class ArcRadiusCard:
    sigma0_m: float = 2.0e-3
    a_I: float = 5.0e-6
    a_V: float = 1.0e-5
    a_ctwd: float = 0.15
    lambda_tier_c: float = 1.0  # enhanced_k_factor
    discrepancy_W_mm: float = 0.3
    discrepancy_D_mm: float = 0.2

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "arc_radius_law": {
                "sigma0_mm": self.sigma0_m * 1e3,
                "a_I": self.a_I,
                "a_V": self.a_V,
                "a_ctwd": self.a_ctwd,
            },
            "tier_c": {"enhanced_k_factor": self.lambda_tier_c},
            "discrepancy_mm": {
                "pool_width": self.discrepancy_W_mm,
                "pool_depth": self.discrepancy_D_mm,
            },
        }


# Parameters fixed from independent literature / materials cards — NEVER fitted.
FIXED_PHYSICAL = {
    "dgamma_dT": -3.5e-4,
    "gamma": 1.2,
    "C_darcy_phys": 1.0e7,
    "C_acc": 0.54,
    "evap_cooling_scale": 1.0,
    "eta": 0.82,  # may be material-card locked
}


def log_likelihood(
    obs: list[dict[str, float]],
    pred: list[dict[str, float]],
    card: ArcRadiusCard,
) -> float:
    ll = 0.0
    for o, p in zip(obs, pred):
        for key, sig in (
            ("pool_width_mm", card.discrepancy_W_mm),
            ("pool_depth_mm", card.discrepancy_D_mm),
        ):
            if key not in o or key not in p:
                continue
            r = (p[key] - o[key]) / max(sig, 1e-6)
            ll -= 0.5 * r * r + math.log(sig)
    return ll


def mcmc(
    log_prob: Callable[[ArcRadiusCard], float],
    start: ArcRadiusCard,
    *,
    n_steps: int = 200,
    seed: int = 0,
) -> list[ArcRadiusCard]:
    rng = random.Random(seed)
    chain = [start]
    cur = start
    lp = log_prob(cur)
    for _ in range(n_steps):
        prop = ArcRadiusCard(
            sigma0_m=max(0.5e-3, cur.sigma0_m * math.exp(rng.gauss(0, 0.05))),
            a_I=cur.a_I + rng.gauss(0, 1e-6),
            a_V=cur.a_V + rng.gauss(0, 2e-6),
            a_ctwd=cur.a_ctwd + rng.gauss(0, 0.02),
            lambda_tier_c=max(1.0, cur.lambda_tier_c + rng.gauss(0, 0.05)),
            discrepancy_W_mm=cur.discrepancy_W_mm,
            discrepancy_D_mm=cur.discrepancy_D_mm,
        )
        lp_p = log_prob(prop)
        if math.log(rng.random() + 1e-30) < lp_p - lp:
            cur, lp = prop, lp_p
        chain.append(cur)
    return chain


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--obs",
        default="validation/baselines/calibration_observations.json",
        help="JSON list of {I,V,CTWD_mm,pool_width_mm,pool_depth_mm,...}",
    )
    ap.add_argument("--out", default="materials/calibration/ER70S-6.global_card.yaml")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parents[1]
    obs_path = root / args.obs
    if not obs_path.exists():
        # Write a stub observation file from bead_calibrate reference
        stub = [
            {
                "I": 180.0,
                "V": 20.0,
                "CTWD_mm": 12.0,
                "pool_width_mm": 6.2,
                "pool_depth_mm": 1.4,
                "held_out": False,
            }
        ]
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_path.write_text(json.dumps(stub, indent=2))
        print(f"[bayesian_calibrate] wrote stub observations → {obs_path}")
    obs = json.loads(obs_path.read_text())
    train = [o for o in obs if not o.get("held_out")]

    def log_prob(card: ArcRadiusCard) -> float:
        # Surrogate: width ~ 2.5*σ_mm + 0.01*I, depth ~ 0.4*σ_mm + 0.002*I
        # (placeholder GP mean until real sims are wired)
        pred = []
        for o in train:
            from waam_twin.physics.heat_source import sigma_arc_m

            sig = sigma_arc_m(
                o["I"], o["V"], o["CTWD_mm"] * 1e-3,
                sigma0_m=card.sigma0_m, a_I=card.a_I, a_V=card.a_V, a_ctwd=card.a_ctwd,
            )
            sig_mm = sig * 1e3
            pred.append(
                {
                    "pool_width_mm": 2.2 * sig_mm * card.lambda_tier_c ** 0.25
                    + 0.005 * o["I"],
                    "pool_depth_mm": 0.45 * sig_mm * card.lambda_tier_c ** 0.35
                    + 0.002 * o["I"],
                }
            )
        return log_likelihood(train, pred, card)

    chain = mcmc(log_prob, ArcRadiusCard(), n_steps=args.steps)
    # Posterior mean of last half
    half = chain[len(chain) // 2 :]
    mean = ArcRadiusCard(
        sigma0_m=sum(c.sigma0_m for c in half) / len(half),
        a_I=sum(c.a_I for c in half) / len(half),
        a_V=sum(c.a_V for c in half) / len(half),
        a_ctwd=sum(c.a_ctwd for c in half) / len(half),
        lambda_tier_c=sum(c.lambda_tier_c for c in half) / len(half),
    )
    out = {
        "card": mean.to_yaml_dict(),
        "fixed_physical": FIXED_PHYSICAL,
        "protocol": "held_out; never retune physical knobs to pass gates",
        "n_mcmc": args.steps,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    out_path.write_text(yaml.safe_dump(out, sort_keys=False))
    print(json.dumps({"wrote": str(out_path), "mean": asdict(mean)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
