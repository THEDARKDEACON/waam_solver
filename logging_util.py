"""Central logging for waam_twin (replaces scattered print())."""

from __future__ import annotations

import logging
import sys

_LOG = logging.getLogger("waam_twin")
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_handler)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def info(msg: str) -> None:
    _LOG.info(msg)


def warning(msg: str) -> None:
    _LOG.warning(msg)
