"""
WAAM Digital Twin v2 — Taichi GPU multiphysics engine for WAAM melt pools.
"""
from .twin import WAAMTwin
from . import platform

__all__ = ["WAAMTwin", "platform"]
__version__ = "2.0.0"
