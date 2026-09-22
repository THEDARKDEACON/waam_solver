"""GPU kernel compiler — Quadrants (independent fork of Taichi).

Kernel source keeps the historical ``ti`` spelling (``@ti.kernel``,
``ti.field``, ``ti.template()``). That is a Quadrants alias, not the
PyPI ``taichi`` package. Fastcache / GPU graphs stay off until the
lock suite is green on this compiler.
"""

from __future__ import annotations

import quadrants as qd

# Kernels and grid code import this name. Quadrants still exposes the
# Taichi-era surface we use: field, kernel, func, template, Vector, ndrange.
ti = qd

__all__ = ["ti", "qd"]
