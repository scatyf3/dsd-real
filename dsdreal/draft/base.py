"""What a drafter must implement, regardless of where it runs."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@runtime_checkable
class DraftEngine(Protocol):
    """Owns draft-side model state and KV for a set of sequences.

    Implementations: EchoDraftEngine (no GPU, for plumbing tests),
    VllmDraftEngine (a real vLLM engine), ReplayDraftEngine (trace replay).
    """

    node_id: str

    def speculate(self, request: SpeculateRequest) -> SpeculateReply: ...

    def free(self, seq_ids: list[str]) -> None: ...

    def close(self) -> None: ...
