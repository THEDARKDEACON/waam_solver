"""
WAAM Digital Twin v2 — Quadrants GPU multiphysics engine for WAAM melt pools.
"""
from .twin import WAAMTwin
from . import runtime

__all__ = ["WAAMTwin", "runtime"]
__version__ = "2.0.0"
