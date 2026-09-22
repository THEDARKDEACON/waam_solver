"""Process-gate percentages.

Local tripwires stay at the README table (calibrated pool 25%, etc.).
Set WAAM_POOL_GATE_PCT to tighten a pool gate without retuning locked
Goldak / η / evap_cooling_scale / C_acc — those knobs are frozen by
assert_physics_lock on held-out jobs.
"""

from __future__ import annotations

import os


def process_gate_pct(default: float, env_key: str = "WAAM_POOL_GATE_PCT") -> float:
    raw = os.environ.get(env_key)
    if raw is None or raw.strip() == "":
        return float(default)
    return float(raw)
