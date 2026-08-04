"""FULL-suite wrapper: run held-out prediction sims (lock + trends)."""

from __future__ import annotations

from waam_twin.validation.test_heldout_prediction import run as _run


def run(**kwargs):
    return _run(run_sims=True, **kwargs)


if __name__ == "__main__":
    import sys
    try:
        _run(run_sims=True)
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
