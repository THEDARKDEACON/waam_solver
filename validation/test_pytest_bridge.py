"""Parametrize the core run_all list for pytest.

Prefer ``python -m waam_twin.validation.run_all`` (single Taichi runtime).
"""

from __future__ import annotations

import pytest

from waam_twin.validation.run_all import core_tests

_CORE = core_tests()


@pytest.mark.parametrize("name,module", _CORE, ids=[n for n, _ in _CORE])
def test_core_validation(name: str, module: str) -> None:
    mod = __import__(module, fromlist=["run"])
    mod.run()
