"""Repository root resolution for the standalone waam_twin project."""

from __future__ import annotations

import pathlib

# Git repo root = directory containing this package (twin.py, jobs/, materials/, …)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
