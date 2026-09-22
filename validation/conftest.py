"""Pytest fixtures for waam_twin.validation.

Prefer ``python -m waam_twin.validation.run_all`` (one Taichi runtime).
Use ``pytest waam_twin/validation/test_pytest_bridge.py`` as the pytest entry.
"""

from __future__ import annotations

import pytest

from waam_twin.runtime import init_taichi


@pytest.fixture(scope="session", autouse=True)
def init_taichi_session():
    init_taichi(backend="cpu")
    yield
