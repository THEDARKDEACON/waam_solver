"""Repository root resolution for the standalone waam_twin project."""

from __future__ import annotations

import pathlib

# Git repo root = directory containing this package (twin.py, jobs/, materials/, …)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


def resolve_project_path(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve relative paths against PROJECT_ROOT (waam_twin/), then cwd."""
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return p
