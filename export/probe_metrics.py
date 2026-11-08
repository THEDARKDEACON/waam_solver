"""Weld thermal-cycle metrics from a probe's T(t).

  peak_C            highest temperature reached
  t_peak_ms         when it was reached
  t85_s             t8/5: time to cool from 800 °C to 500 °C on the last
                    cycle that went above 800 °C (NaN until it has cooled to
                    500 °C). The standard single number for HAZ microstructure.
  above_solidus_ms  total time above the solidus (the probe point was melted)
  cool_700_K_s      cooling rate as the last cycle passes 700 °C [K/s]
  T_now_C           latest reading

Crossings are linearly interpolated between samples. For a multi-layer wall
every layer reheats the probe; "last cycle" is the most recent pass through
800 °C, which is the one that sets the final microstructure.
"""

from __future__ import annotations

import math

import numpy as np

C0 = 273.15
_NAN = float("nan")


def _down_crossing(t: np.ndarray, T: np.ndarray, level: float, start: int) -> float:
    """Time of the first downward crossing of ``level`` at or after ``start``."""
    for i in range(max(start, 1), len(T)):
        if T[i - 1] >= level > T[i]:
            f = (T[i - 1] - level) / (T[i - 1] - T[i])
            return float(t[i - 1] + f * (t[i] - t[i - 1]))
    return _NAN


def thermal_cycle_metrics(t_ms, T_K, T_solidus_K: float) -> dict:
    t = np.asarray(t_ms, dtype=np.float64)
    T = np.asarray(T_K, dtype=np.float64)
    out = {
        "peak_C": _NAN, "t_peak_ms": _NAN, "t85_s": _NAN,
        "above_solidus_ms": 0.0, "cool_700_K_s": _NAN, "T_now_C": _NAN,
    }
    if T.size == 0:
        return out
    ip = int(np.argmax(T))
    out["peak_C"] = round(float(T[ip]) - C0, 1)
    out["t_peak_ms"] = round(float(t[ip]), 3)
    out["T_now_C"] = round(float(T[-1]) - C0, 1)
    if T.size >= 2:
        hot = (T[:-1] >= T_solidus_K) & (T[1:] >= T_solidus_K)
        out["above_solidus_ms"] = round(float(np.sum(np.diff(t)[hot])), 3)
    T800, T700, T500 = 800.0 + C0, 700.0 + C0, 500.0 + C0
    above = np.nonzero(T >= T800)[0]
    if above.size:
        # Start of the last excursion above 800 °C.
        last = int(above[-1])
        first = last
        while first > 0 and T[first - 1] >= T800:
            first -= 1
        t8 = _down_crossing(t, T, T800, first)
        if math.isfinite(t8):
            i8 = int(np.searchsorted(t, t8))
            t5 = _down_crossing(t, T, T500, i8)
            if math.isfinite(t5):
                out["t85_s"] = round((t5 - t8) / 1000.0, 4)
            for i in range(max(i8, 1), len(T)):
                if T[i - 1] >= T700 > T[i] and t[i] > t[i - 1]:
                    out["cool_700_K_s"] = round(float((T[i - 1] - T[i]) / ((t[i] - t[i - 1]) / 1000.0)), 1)
                    break
    return out
