"""Feasibility constraints for discrete suffix search.

The search objective may combine route and content terms.  A route floor keeps
the content optimizer from selecting a candidate that looks good only because
it has destroyed the modality gateway.  This module is deliberately tensor
agnostic so the policy can be unit-tested without loading a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RouteFloor:
    """Maximum allowed BOI cross-entropy; lower values are better."""

    max_boi_ce: float

    def __post_init__(self) -> None:
        value = float(self.max_boi_ce)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("max_boi_ce must be a finite non-negative number")

    def accepts(self, boi_ce: float) -> bool:
        """Return whether a candidate satisfies the route constraint."""
        value = float(boi_ce)
        return math.isfinite(value) and value <= self.max_boi_ce


def route_floor_from_optional(value: float | None) -> RouteFloor | None:
    """Build a route floor from a CLI value, preserving ``None`` semantics."""
    return None if value is None else RouteFloor(float(value))
