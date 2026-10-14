import math

import pytest

from morph.suffix.constraints import RouteFloor, route_floor_from_optional


def test_route_floor_accepts_lower_or_equal_ce():
    floor = RouteFloor(1.5)
    assert floor.accepts(1.5)
    assert floor.accepts(0.2)
    assert not floor.accepts(1.5001)


def test_route_floor_rejects_nonfinite_values():
    floor = RouteFloor(1.5)
    assert not floor.accepts(math.inf)
    assert not floor.accepts(math.nan)


def test_route_floor_validation_and_optional_factory():
    assert route_floor_from_optional(None) is None
    assert route_floor_from_optional(2.0) == RouteFloor(2.0)
    with pytest.raises(ValueError):
        RouteFloor(-1.0)
    with pytest.raises(ValueError):
        RouteFloor(math.inf)
