"""Streamline tracing for the interactive viewer."""

from __future__ import annotations

import numpy as np


def trace_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    seeds: list[tuple[int, int, int]],
    n_steps: int = 40,
    step_cells: float = 0.65,
    min_speed_lu: float = 1e-7,
) -> list[np.ndarray]:
    """
    Trace streamlines in cell-index space.

    Each step advances ``step_cells`` along the normalized velocity direction
    so lines remain visible (raw lattice u is ~O(0.01) per cell).
    """
    nx, ny, nz = ux.shape
    lines: list[np.ndarray] = []
    for si, sj, sk in seeds:
        p = np.array([float(si), float(sj), float(sk)], dtype=np.float64)
        pts = [p.copy()]
        for _ in range(n_steps):
            i = int(np.clip(round(p[0]), 0, nx - 1))
            j = int(np.clip(round(p[1]), 0, ny - 1))
            k = int(np.clip(round(p[2]), 0, nz - 1))
            u = np.array([ux[i, j, k], uy[i, j, k], uz[i, j, k]], dtype=np.float64)
            umag = float(np.linalg.norm(u))
            if umag < min_speed_lu:
                break
            p = p + (u / umag) * step_cells
            if (
                p[0] < 0.5
                or p[0] >= nx - 0.5
                or p[1] < 0.5
                or p[1] >= ny - 0.5
                or p[2] < 0.5
                or p[2] >= nz - 0.5
            ):
                break
            pts.append(p.copy())
        if len(pts) >= 2:
            lines.append(np.array(pts))
    return lines


def seeds_near_torch(
    nx: int,
    ny: int,
    nz: int,
    ti: int,
    tj: int,
    tk: int,
    n: int = 16,
) -> list[tuple[int, int, int]]:
    """Seed points in a small box above the torch cell."""
    seeds: list[tuple[int, int, int]] = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        di = int(rng.integers(-2, 3))
        dj = int(rng.integers(-2, 3))
        dk = int(rng.integers(0, 4))
        seeds.append((
            int(np.clip(ti + di, 0, nx - 1)),
            int(np.clip(tj + dj, 0, ny - 1)),
            int(np.clip(tk + dk, 0, nz - 1)),
        ))
    return seeds
