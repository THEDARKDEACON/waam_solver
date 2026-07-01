"""NumPy streamline tracing for the interactive viewer."""

from __future__ import annotations

import numpy as np


def trace_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    seeds: list[tuple[int, int, int]],
    n_steps: int = 30,
) -> list[np.ndarray]:
    """Euler trace in cell-index space (u is lattice velocity)."""
    nx, ny, nz = ux.shape
    lines: list[np.ndarray] = []
    for si, sj, sk in seeds:
        p = np.array([float(si), float(sj), float(sk)], dtype=np.float64)
        pts = [p.copy()]
        for _ in range(n_steps):
            i = int(np.clip(p[0], 0, nx - 1))
            j = int(np.clip(p[1], 0, ny - 1))
            k = int(np.clip(p[2], 0, nz - 1))
            u = np.array([ux[i, j, k], uy[i, j, k], uz[i, j, k]], dtype=np.float64)
            if np.linalg.norm(u) < 1e-8:
                break
            p = p + u
            if p[0] < 0 or p[0] >= nx - 1 or p[1] < 0 or p[1] >= ny - 1 or p[2] < 0 or p[2] >= nz - 1:
                break
            pts.append(p.copy())
        if len(pts) >= 2:
            lines.append(np.array(pts))
    return lines


def seeds_near_torch(nx: int, ny: int, nz: int, ti: int, tj: int, tk: int, n: int = 16) -> list[tuple[int, int, int]]:
    seeds = []
    for _ in range(n):
        di = int(np.random.randint(-2, 3))
        dj = int(np.random.randint(-2, 3))
        dk = int(np.random.randint(0, 3))
        seeds.append((
            int(np.clip(ti + di, 0, nx - 1)),
            int(np.clip(tj + dj, 0, ny - 1)),
            int(np.clip(tk + dk, 0, nz - 1)),
        ))
    return seeds
