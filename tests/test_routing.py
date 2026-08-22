"""RoutingPolicy implementations, in isolation from RemoteDraftProposer."""
from __future__ import annotations

import pytest

from dsdreal.policy.routing import RoundRobin


def test_round_robin_cycles_in_order():
    rr = RoundRobin()
    picks = [rr.select_drafter(["d0", "d1", "d2"]) for _ in range(7)]
    assert picks == ["d0", "d1", "d2", "d0", "d1", "d2", "d0"]


def test_round_robin_single_drafter_is_a_no_op():
    rr = RoundRobin()
    assert [rr.select_drafter(["d0"]) for _ in range(3)] == ["d0"] * 3


def test_round_robin_rejects_empty_registry():
    with pytest.raises(ValueError, match="no drafters"):
        RoundRobin().select_drafter([])
