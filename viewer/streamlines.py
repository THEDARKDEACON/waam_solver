"""Streamline tracing for the interactive viewer."""

from __future__ import annotations

import numpy as np


def trace_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    seeds: list[tuple[int, int, int]],
    n_steps: int = 48,
    step_cells: float = 0.55,
    min_speed_lu: float = 1e-6,
    f_l: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    flag_gas: int = 2,
) -> list[np.ndarray]:
    """
    Trace streamlines in cell-index space.

    Advances a fixed step along the local velocity direction so lines stay
    visible even when lattice |u| is O(0.01). Stops in gas / dry cells.
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
            if flags is not None and int(flags[i, j, k]) == flag_gas:
                break
            if f_l is not None and float(f_l[i, j, k]) < 0.08:
                break
            u = np.array(
                [float(ux[i, j, k]), float(uy[i, j, k]), float(uz[i, j, k])],
                dtype=np.float64,
            )
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
        if len(pts) >= 3:
            lines.append(np.asarray(pts, dtype=np.float32))
    return lines


def seeds_in_liquid_near_torch(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    f_l: np.ndarray,
    flags: np.ndarray,
    ti: int,
    tj: int,
    tk: int,
    flag_gas: int,
    n: int = 24,
    search_r: int = 8,
    min_speed_lu: float = 1e-6,
) -> list[tuple[int, int, int]]:
    """
    Pick liquid cells with nonzero |u| near the torch.

    Falls back to any liquid near the torch, then to a small geometric box.
    """
    nx, ny, nz = f_l.shape
    i0 = int(np.clip(ti, 0, nx - 1))
    j0 = int(np.clip(tj, 0, ny - 1))
    k0 = int(np.clip(tk, 0, nz - 1))

    candidates: list[tuple[float, int, int, int]] = []
    for i in range(max(0, i0 - search_r), min(nx, i0 + search_r + 1)):
        for j in range(max(0, j0 - search_r), min(ny, j0 + search_r + 1)):
            for k in range(max(0, k0 - search_r), min(nz, k0 + search_r + 1)):
                if int(flags[i, j, k]) == flag_gas:
                    continue
                if float(f_l[i, j, k]) < 0.25:
                    continue
                speed = float(np.sqrt(
                    ux[i, j, k] ** 2 + uy[i, j, k] ** 2 + uz[i, j, k] ** 2
                ))
                if speed < min_speed_lu:
                    continue
                dist = (i - i0) ** 2 + (j - j0) ** 2 + (k - k0) ** 2
                # Prefer fast flow near torch
                score = speed / (1.0 + 0.15 * dist)
                candidates.append((score, i, j, k))

    if not candidates:
        # Liquid only (even if nearly stagnant) — still seed so lines can grow
        for i in range(max(0, i0 - search_r), min(nx, i0 + search_r + 1)):
            for j in range(max(0, j0 - search_r), min(ny, j0 + search_r + 1)):
                for k in range(max(0, k0 - 2), min(nz, k0 + search_r + 1)):
                    if int(flags[i, j, k]) == flag_gas:
                        continue
                    if float(f_l[i, j, k]) < 0.5:
                        continue
                    candidates.append((1.0, i, j, k))

    if not candidates:
        return [
            (
                int(np.clip(i0 + di, 0, nx - 1)),
                int(np.clip(j0 + dj, 0, ny - 1)),
                int(np.clip(k0 + dk, 0, nz - 1)),
            )
            for di, dj, dk in (
                (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (2, 1, 0), (-2, -1, 1),
            )
        ]

    candidates.sort(reverse=True)
    # Spread seeds: take top scores with a mild stride
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for _, i, j, k in candidates:
        key = (i, j, k)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= n:
            break
    return out


# Back-compat alias used by older imports
def seeds_near_torch(
    nx: int,
    ny: int,
    nz: int,
    ti: int,
    tj: int,
    tk: int,
    n: int = 16,
) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    seeds: list[tuple[int, int, int]] = []
    for _ in range(n):
        di = int(rng.integers(-3, 4))
        dj = int(rng.integers(-3, 4))
        dk = int(rng.integers(0, 5))
        seeds.append((
            int(np.clip(ti + di, 0, nx - 1)),
            int(np.clip(tj + dj, 0, ny - 1)),
            int(np.clip(tk + dk, 0, nz - 1)),
        ))
    return seeds
