"""How a target reaches its drafters.

`propose()` runs synchronously inside the vLLM engine step, so this API is
blocking by construction — no asyncio anywhere on this path.

The batch contract matters: `speculate()` takes the whole step's requests at
once so an implementation can fan out concurrently. Wall clock must be
max(RTT) over drafters, never the sum; a per-request API would silently
serialise N draft nodes and destroy the thing we are trying to measure.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

from dsdreal.netem.base import NetworkEmulator, block_for
from dsdreal.netem.null import NullEmulator
from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


class DraftClient(ABC):
    """Target-side handle on one or more drafters."""

    def __init__(self, node_id: str = "t0",
                 emulator: NetworkEmulator | None = None) -> None:
        self.node_id = node_id
        self.emulator = emulator or NullEmulator()

    @abstractmethod
    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        """Replies in the same order as `requests`."""

    @abstractmethod
    def free(self, seq_ids: Iterable[str]) -> None: ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class InProcessDraftClient(DraftClient):
    """Simplest possible transport: call the drafter directly.

    No sockets, no serialization. Used by Stage 2 (fused mode) and by every
    no-GPU test. Swapping this for the ZMQ client is the whole of Stage 3 —
    nothing above this interface changes.

    Delay is still applied, so fused runs produce comparable timings. Because
    the calls are inevitably serial here, the emulated delay is charged once
    per direction as max() over the batch, mirroring what a concurrent
    fan-out would cost rather than what a serial loop would.
    """

    def __init__(self, engines, node_id: str = "t0", emulator=None) -> None:
        super().__init__(node_id=node_id, emulator=emulator)
        if not isinstance(engines, dict):
            engines = {engines.node_id: engines}
        self.engines = engines

    def _engine_for(self, drafter_id: str):
        try:
            return self.engines[drafter_id]
        except KeyError:
            raise KeyError(
                f"unknown drafter_id {drafter_id!r}; "
                f"registered: {sorted(self.engines)}"
            ) from None

    def speculate(self, requests: list[SpeculateRequest]) -> list[SpeculateReply]:
        if not requests:
            return []
        sent = time.perf_counter()
        out_ms = [self.emulator.one_way_ms(self.node_id, r.drafter_id)
                  for r in requests]
        block_for(max(out_ms) / 1000.0)

        replies = [self._engine_for(r.drafter_id).speculate(r) for r in requests]

        back_ms = [self.emulator.one_way_ms(r.drafter_id, self.node_id)
                   for r in requests]
        block_for(max(back_ms) / 1000.0)
        recv = time.perf_counter()

        for reply, o, b in zip(replies, out_ms, back_ms):
            reply.sent_ms = sent * 1000.0
            reply.recv_ms = recv * 1000.0
            reply.injected_delay_ms = o + b
        return replies

    def free(self, seq_ids: Iterable[str]) -> None:
        ids = list(seq_ids)
        for engine in self.engines.values():
            engine.free(ids)

    def close(self) -> None:
        for engine in self.engines.values():
            engine.close()


def make_client(kind: str, **kwargs) -> DraftClient:
    """Factory so config can name a transport without importing it."""
    if kind in ("inprocess", "local", "fused"):
        return InProcessDraftClient(**kwargs)
    if kind == "zmq":
        raise NotImplementedError("ZmqDraftClient lands in Stage 3")
    raise ValueError(f"unknown transport {kind!r}")
