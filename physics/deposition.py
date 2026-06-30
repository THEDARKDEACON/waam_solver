"""Wire deposition and droplet momentum."""

from __future__ import annotations

from .. import kernels

feed_wire = kernels.feed_wire
feed_wire_momentum = kernels.feed_wire_momentum

__all__ = ["feed_wire", "feed_wire_momentum"]
