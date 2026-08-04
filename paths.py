"""Repository root resolution and path sandboxing for waam_twin."""

from __future__ import annotations

import os
import pathlib

# Git repo root = directory containing this package (twin.py, jobs/, materials/, …)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


def _allow_absolute_outside_root() -> bool:
    return os.environ.get("WAAM_ALLOW_ABSOLUTE_PATHS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def is_under_root(path: pathlib.Path, root: pathlib.Path | None = None) -> bool:
    """Return True if ``path`` resolves inside ``root`` (default PROJECT_ROOT)."""
    root = (root or PROJECT_ROOT).resolve()
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def resolve_project_path(
    path: str | pathlib.Path,
    *,
    must_exist: bool = False,
    allow_outside: bool | None = None,
) -> pathlib.Path:
    """Resolve a path against PROJECT_ROOT with optional sandboxing.

    Relative paths are resolved under PROJECT_ROOT (falling back to the
    given relative path if the candidate does not exist and ``must_exist``
    is False — matching historical cwd behaviour for tooling).

    Absolute paths outside PROJECT_ROOT are rejected unless
    ``allow_outside`` is True or ``WAAM_ALLOW_ABSOLUTE_PATHS=1``.
    """
    p = pathlib.Path(path)
    if allow_outside is None:
        allow_outside = _allow_absolute_outside_root()

    if p.is_absolute():
        resolved = p.resolve()
        if not allow_outside and not is_under_root(resolved):
            raise ValueError(
                f"Absolute path outside project root refused: {resolved} "
                f"(root={PROJECT_ROOT}). Relocate under the repo or set "
                "WAAM_ALLOW_ABSOLUTE_PATHS=1."
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    candidate = (PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return candidate
    if must_exist:
        raise FileNotFoundError(f"{path} (tried {candidate})")
    # Historical fallback: treat as cwd-relative when not under PROJECT_ROOT.
    cwd_cand = p.resolve()
    if cwd_cand.exists():
        if not allow_outside and not is_under_root(cwd_cand):
            raise ValueError(
                f"Path resolves outside project root: {cwd_cand} "
                f"(root={PROJECT_ROOT}). Use a path under the repo or set "
                "WAAM_ALLOW_ABSOLUTE_PATHS=1."
            )
        return cwd_cand
    return candidate


def resolve_output_path(
    path: str | pathlib.Path,
    *,
    allow_outside: bool | None = None,
) -> pathlib.Path:
    """Resolve an output/write path; create parent dirs are caller's job.

    Same sandbox rules as :func:`resolve_project_path`.
    """
    return resolve_project_path(path, must_exist=False, allow_outside=allow_outside)
