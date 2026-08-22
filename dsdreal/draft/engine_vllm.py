"""Draft engine backed by a real vLLM model.

Stage 2a scope (option A from the design discussion): stateless per-round
`generate()` over the full confirmed prefix, relying on `enable_prefix_caching`
(a stable, public engine flag) to make repeated calls over a growing prefix
cheap. No vLLM internals are touched -- deliberately, since anything more
"efficient" (a persistent, pausable, rollback-aware draft-side session) would
mean reimplementing the target's own KV-rollback scheduling against
version-sensitive internals for a real drafter that, on every rejection round,
would have to fall back to resubmission anyway. See the Stage 2a design
discussion for the full argument.

Correctness rests entirely on determinism: fp32 + `enforce_eager` + greedy
(temperature=0) makes every `generate()` call over the same token prefix
reproduce the same continuation regardless of whether it hit the prefix
cache, so state here is just bookkeeping, never a source of divergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dsdreal.proto.messages import SpeculateReply, SpeculateRequest


@dataclass
class _SeqState:
    confirmed: list[int] = field(default_factory=list)


class VllmDraftEngine:
    """Wraps an already-constructed `vllm.LLM` (the caller owns its
    lifecycle -- sizing/dtype/gpu_memory_utilization are a construction-time
    decision made once, not this class's concern)."""

    def __init__(self, llm, node_id: str = "d0") -> None:
        self.node_id = node_id
        self.llm = llm
        self.state: dict[str, _SeqState] = {}
        self.calls = 0

    def speculate(self, request: SpeculateRequest) -> SpeculateReply:
        self.calls += 1
        seq = self.state.get(request.seq_id)
        if seq is None:
            seq = _SeqState(confirmed=list(request.prompt_token_ids or []))
            self.state[request.seq_id] = seq

        seq.confirmed.extend(request.accepted_tokens)
        if request.num_confirmed and len(seq.confirmed) != request.num_confirmed:
            raise ValueError(
                f"drafter desync on {request.seq_id}: local "
                f"{len(seq.confirmed)} != target {request.num_confirmed}"
            )

        if request.gamma <= 0:
            return SpeculateReply(seq_id=request.seq_id, draft_tokens=[])

        import time

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        t0 = time.perf_counter()
        (out,) = self.llm.generate(
            [TokensPrompt(prompt_token_ids=seq.confirmed)],
            SamplingParams(temperature=0.0, max_tokens=request.gamma),
            use_tqdm=False,
        )
        tokens = list(out.outputs[0].token_ids)
        return SpeculateReply(
            seq_id=request.seq_id,
            draft_tokens=tokens,
            draft_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def free(self, seq_ids: list[str]) -> None:
        for sid in seq_ids:
            self.state.pop(sid, None)

    def close(self) -> None:
        self.state.clear()
