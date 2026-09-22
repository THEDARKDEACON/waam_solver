"""D3Q19 velocity set — single source for grid fields and kernel immediates."""

from __future__ import annotations

Q = 19

EX = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
EY = (0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1)
EZ = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1)
W = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)
OPP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)

# Core SoA volume fields always allocated by WAAMGrid (see grid.py).
N_CORE_VOLUME = 27
# flags counted separately in estimated_vram_mb
N_FLAGS = 1
N_VOF_OPTIONAL = 1
N_LORENTZ_OPTIONAL = 10
N_EXPORT_OPTIONAL = 2
N_ELEC_BINS = 64


def estimate_fields_vram_mb(
    nx: int,
    ny: int,
    nz: int,
    max_tracers: int,
    *,
    lorentz: bool = True,
    vof: bool = True,
    export: bool = True,
) -> float:
    """Host-side VRAM estimate matching WAAMGrid.estimated_vram_mb."""
    n = int(nx) * int(ny) * int(nz)
    dist = 2 * Q * n * 4
    n_vol = N_CORE_VOLUME + N_FLAGS
    if vof:
        n_vol += N_VOF_OPTIONAL
    if lorentz:
        n_vol += N_LORENTZ_OPTIONAL
    if export:
        n_vol += N_EXPORT_OPTIONAL
    volume = n_vol * n * 4
    bins = nz * N_ELEC_BINS * 4 if lorentz else 0
    slice_buf = ny * nz * 4
    tracers = int(max_tracers) * (3 * 4 + 4)
    return (dist + volume + bins + slice_buf + tracers) / (1024 ** 2)
