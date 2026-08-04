"""Streamline tracing for the interactive viewer."""

from __future__ import annotations

import numpy as np


def _trace_one_direction(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    seed: tuple[int, int, int],
    *,
    sign: float,
    n_steps: int,
    step_cells: float,
    min_speed_lu: float,
    f_l: np.ndarray | None,
    flags: np.ndarray | None,
    flag_gas: int,
    fl_cut: float,
) -> list[np.ndarray]:
    nx, ny, nz = ux.shape
    p = np.array([float(seed[0]), float(seed[1]), float(seed[2])], dtype=np.float64)
    pts = [p.copy()]
    for _ in range(n_steps):
        i = int(np.clip(round(p[0]), 0, nx - 1))
        j = int(np.clip(round(p[1]), 0, ny - 1))
        k = int(np.clip(round(p[2]), 0, nz - 1))
        if flags is not None and int(flags[i, j, k]) == flag_gas:
            break
        if f_l is not None and float(f_l[i, j, k]) < fl_cut:
            break
        u = np.array(
            [float(ux[i, j, k]), float(uy[i, j, k]), float(uz[i, j, k])],
            dtype=np.float64,
        )
        umag = float(np.linalg.norm(u))
        if umag < min_speed_lu:
            break
        p = p + sign * (u / umag) * step_cells
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
    return pts


def trace_streamlines(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    seeds: list[tuple[int, int, int]],
    n_steps: int = 140,
    step_cells: float = 0.4,
    min_speed_lu: float = 5e-7,
    f_l: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    flag_gas: int = 2,
    fl_cut: float = 0.04,
    bidirectional: bool = True,
) -> list[np.ndarray]:
    """
    Trace streamlines in cell-index space.

    Bidirectional integration (upstream + downstream) makes lines much longer
    than forward-only traces that die at the freeze front.
    """
    lines: list[np.ndarray] = []
    for seed in seeds:
        fwd = _trace_one_direction(
            ux, uy, uz, seed,
            sign=1.0, n_steps=n_steps, step_cells=step_cells,
            min_speed_lu=min_speed_lu, f_l=f_l, flags=flags,
            flag_gas=flag_gas, fl_cut=fl_cut,
        )
        if bidirectional:
            bwd = _trace_one_direction(
                ux, uy, uz, seed,
                sign=-1.0, n_steps=n_steps, step_cells=step_cells,
                min_speed_lu=min_speed_lu, f_l=f_l, flags=flags,
                flag_gas=flag_gas, fl_cut=fl_cut,
            )
            # upstream (reversed, drop seed) + seed + downstream
            merged = list(reversed(bwd[1:])) + fwd
        else:
            merged = fwd
        if len(merged) >= 4:
            lines.append(np.asarray(merged, dtype=np.float32))
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
    n: int = 36,
    search_r: int = 14,
    min_speed_lu: float = 5e-7,
) -> list[tuple[int, int, int]]:
    """Pick liquid cells with nonzero |u| near the torch (spread in the pool)."""
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
                if float(f_l[i, j, k]) < 0.15:
                    continue
                speed = float(np.sqrt(
                    ux[i, j, k] ** 2 + uy[i, j, k] ** 2 + uz[i, j, k] ** 2
                ))
                if speed < min_speed_lu:
                    continue
                dist = (i - i0) ** 2 + (j - j0) ** 2 + (k - k0) ** 2
                score = speed / (1.0 + 0.08 * dist)
                candidates.append((score, i, j, k))

    if not candidates:
        for i in range(max(0, i0 - search_r), min(nx, i0 + search_r + 1)):
            for j in range(max(0, j0 - search_r), min(ny, j0 + search_r + 1)):
                for k in range(max(0, k0 - 2), min(nz, k0 + search_r + 1)):
                    if int(flags[i, j, k]) == flag_gas:
                        continue
                    if float(f_l[i, j, k]) < 0.35:
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
                (0, 0, 1), (2, 1, 0), (-2, -1, 1), (3, 0, 1), (-3, 2, 0),
            )
        ]

    candidates.sort(reverse=True)
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    # Spatial thinning so seeds are not all on the same cell
    min_sep2 = 2.25
    for _, i, j, k in candidates:
        key = (i, j, k)
        if key in seen:
            continue
        if any((i - a) ** 2 + (j - b) ** 2 + (k - c) ** 2 < min_sep2 for a, b, c in out):
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= n:
            break
    if len(out) < max(8, n // 3):
        # Fill without spacing if pool is tiny
        for _, i, j, k in candidates:
            if (i, j, k) not in seen:
                out.append((i, j, k))
                seen.add((i, j, k))
            if len(out) >= n:
                break
    return out


def seeds_near_torch(
    nx: int,
    ny: int,
    nz: int,
    ti: int,
    tj: int,
    tk: int,
    n: int = 16,
) -> list[tuple[int, int, int]]:
    """Legacy geometric seed box (no velocity field)."""
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
