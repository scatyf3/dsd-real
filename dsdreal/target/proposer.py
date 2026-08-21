"""`custom_class` speculative proposer: turns vLLM's propose() calls into
draft-client requests instead of running a draft model.

Registered via vLLM's native `method="custom_class"` plug point
(vllm/v1/spec_decode/custom_class_proposer.py, confirmed present in the
0.27.1 pin -- see docs/stage0-findings.md K6). vLLM imports the class named
in `speculative_config.model`, instantiates it with only `vllm_config`, and
calls `propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
slot_mappings=None)` once per engine step (gpu_model_runner.py, the
`method == "custom_class"` branch). No NgramProposer subclassing, no
isinstance dispatch assert, and no constructor warm-up call to defend
against -- those are quirks of the ngram dispatch path this one doesn't
share.

`num_tokens_no_spec[i]` is slot i's confirmed length: prompt plus every
accepted/bonus token so far, already including whatever was just sampled
this step (NgramProposer itself relies on this to search its own history for
matches). `token_ids_cpu[i, :num_tokens_no_spec[i]]` is that history.

Stage 0 has no transport-level request id yet, so identity is recovered
positionally: a slot's `num_tokens_no_spec` only grows while the same
request occupies it, so a same-or-larger value means "still the request we
last saw here" and a smaller one means the slot was handed to someone new.
"""
from __future__ import annotations

from dsdreal.proto.transport import DraftClient
from dsdreal.proto.messages import SpeculateRequest

_default_client: DraftClient | None = None


def configure(client: DraftClient) -> None:
    """Call before constructing the LLM.

    `custom_class` instantiates the proposer with only `vllm_config`
    (create_custom_proposer in vLLM), so a module-level default is the only
    way to hand it a transport.
    """
    global _default_client
    _default_client = client


class RemoteDraftProposer:
    def __init__(self, vllm_config, client: DraftClient | None = None,
                 drafter_id: str = "d0") -> None:
        spec_config = vllm_config.speculative_config
        self.k = spec_config.num_speculative_tokens
        self.drafter_id = drafter_id
        self.client = client if client is not None else _default_client
        if self.client is None:
            raise RuntimeError(
                "RemoteDraftProposer needs a DraftClient; call "
                "dsdreal.target.proposer.configure(client) before "
                "constructing the LLM."
            )
        self._slot_seq: dict[int, str] = {}
        self._slot_n_hist: dict[int, int] = {}
        self._next_seq = 0

    def _seq_id_for(self, i: int, n_hist: int) -> tuple[str, bool]:
        """Returns (seq_id, is_fresh)."""
        last = self._slot_n_hist.get(i)
        fresh = last is None or n_hist < last
        if fresh:
            seq_id = f"slot{i}-{self._next_seq}"
            self._next_seq += 1
            self._slot_seq[i] = seq_id
        self._slot_n_hist[i] = n_hist
        return self._slot_seq[i], fresh

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None) -> list[list[int]]:
        requests: list[SpeculateRequest] = []
        active_rows: list[int] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                continue
            n_hist = int(num_tokens_no_spec[i])
            seq_id, fresh = self._seq_id_for(i, n_hist)
            requests.append(SpeculateRequest(
                seq_id=seq_id,
                drafter_id=self.drafter_id,
                gamma=self.k,
                num_confirmed=n_hist,
                prompt_token_ids=(
                    [int(t) for t in token_ids_cpu[i, :n_hist]] if fresh else None
                ),
            ))
            active_rows.append(i)

        out: list[list[int]] = [[] for _ in sampled_token_ids]
        if not requests:
            return out

        replies = self.client.speculate(requests)
        for row, reply in zip(active_rows, replies):
            out[row] = list(reply.draft_tokens)
        return out
