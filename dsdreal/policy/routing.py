"""Which drafter a fresh request gets sent to.

Decided once, at the "fresh" row in RemoteDraftProposer.propose() (a new
request occupying a slot), and stuck to for that request's whole lifetime.
A drafter tracks confirmed length per seq_id (VllmDraftEngine, EchoDraftEngine
-- see their `num_confirmed` desync check); switching drafters mid-request
would hand a later round to a drafter that never saw the earlier ones.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RoutingPolicy(Protocol):
    def select_drafter(self, drafter_ids: list[str]) -> str: ...


class RoundRobin:
    """Cycles through `drafter_ids` in the order given. Stateful only in
    that one counter, so N requests arriving together (a single propose()
    batch) still spread round-robin rather than piling onto drafter_ids[0]."""

    def __init__(self) -> None:
        self._next = 0

    def select_drafter(self, drafter_ids: list[str]) -> str:
        if not drafter_ids:
            raise ValueError("no drafters to route to")
        drafter_id = drafter_ids[self._next % len(drafter_ids)]
        self._next += 1
        return drafter_id
