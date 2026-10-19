# dsd-real

[![CI](https://github.com/scatyf3/dsd-real/actions/workflows/ci.yml/badge.svg)](https://github.com/scatyf3/dsd-real/actions/workflows/ci.yml)

**English** | [中文](README.zh-CN.md)

A real-system implementation of disaggregated speculative decoding (DSD), built to validate the conclusions of the DSD-Sim simulator end to end.

The approach is to **plug into vLLM, not fork it**: the drafter and the target each run their own vLLM engine, and vLLM's native
`speculative_config.method = "custom_class"` slots a remote drafter into the target's speculative path.
All seven Stage 0 probes pass on the pinned platform, and both legs of Stage 2 (fused and distributed) have been verified on real GPUs.

Platform: **vLLM 0.27.1**, pinned in [requirements/gpu.txt](requirements/gpu.txt). Stages 0–2 were verified on 8×L40S (sm_89, 48 GB), and the small-scale Table 3 ran on 4×RTX A6000 (NYULAB).

---

## Architecture

```
   Draft vLLM Server                Draft Client                        Target vLLM Server
 ┌────────────────────┐      ┌──────────────────────┐              ┌──────────────────────────┐
 │ VllmDraftEngine    │      │ ┌─────┐┌────┐┌─────┐ │  ┌────────┐  │ ┌────────────┐ ┌───────┐ │
 │  1. Propose: γ tok │◄────►│ │ IPC ││ZMQ ││HTTP │ │◄►│Routing │◄►│ │Draft Runner│ │Target │ │
 │  2. Rollback:      │      │ └─────┘└────┘└─────┘ │  │ policy │  │ │ [custom_   │ │Runner │ │
 │     re-prefill     │      │  one DraftClient API │  └────────┘  │ │  class]    │ │       │ │
 └────────────────────┘      └──────────────────────┘  RoundRobin  │ └────────────┘ └───────┘ │
   engine_vllm.py ✅          transport.py: all 3 ✅               │   RemoteDraftProposer    │
   engine_replay.py ✅        server.py / http_server.py ✅        │   .propose() once/step   │
   engine_echo.py   ✅        policy/routing.py ✅                 └──────────────────────────┘
                                                                         proposer.py ✅
```

The control loop lives on the **target** side: on every engine step, vLLM itself calls
`RemoteDraftProposer.propose()`, which fans out to the drafters synchronously, waits for all of them, and returns a
variable-length list of draft tokens per request.

**Three contracts (pinned down by code and tests, not by convention):**

- `propose()` runs inside an engine step, **with no event loop** → the whole `DraftClient` path is blocking.
- `speculate()` takes the whole batch at once. It is **not** a per-request API. Wall-clock time has to be `max(RTT_i)`, not
  `Σ RTT_i`. A per-request API would quietly serialize the N draft nodes and wreck the very thing we want to measure.
- Accept/reject feedback **needs no separate channel**: the `num_tokens_no_spec[i]` argument to `propose()` is the
  confirmed length of slot i, and its difference from the previous round is how many tokens were accepted this round (including the bonus token).
  `configure(stats=[])` exports it directly, so there is no need to infer it from timing slopes.

Request identity is currently recovered from slot position (`num_tokens_no_spec` only grows while the same request occupies that slot),
because the protocol does not carry a request id yet.

---

## Transport layer: three implementations, one contract

This is the layer `docs/notes/context.md` describes as "wrap ZMQ-to-HTTP behind an abstraction first". All three implementations sit behind
[proto/transport.py](dsdreal/proto/transport.py), and `make_client(kind, ...)` picks one by name:

| | `InProcessDraftClient` | `ZmqDraftClient` | `HttpDraftClient` |
|---|---|---|---|
| Wire | Direct function call | DEALER/ROUTER + msgpack | HTTP/1.1 + JSON |
| Server | None | [`DraftServer`](dsdreal/draft/server.py) | [`HttpDraftServer`](dsdreal/draft/http_server.py) |
| Concurrency | N/A (serial, latency counted as the max) | One DEALER pipelines many messages: send-all → recv-all | Grouped by drafter, **one POST per drafter**, concurrent via a thread pool |
| Round trips per step | 0 | 1 per drafter (pipelined) | 1 per drafter |
| Error propagation | Raised directly | `ERR ` frame → re-raised on the client | HTTP 400 + `{"error"}` → re-raised on the client |
| Status | ✅ Stage 2a verified on GPU | ✅ Stage 2b verified on GPU | ✅ loopback tests pass, **not yet run on GPU** |

**We keep both.** The number that decides between them is `SpeculateReply.net_excess_ms`
(= measured RTT − injected delay − drafter compute), i.e. the real cost of serialization plus queuing:
HTTP is easier to debug (you can hit it with curl, packet captures are readable, other languages need no protocol change), while ZMQ has lower overhead.
Both assert `net_excess_ms < 20ms` on loopback. Which one wins on a real link is for this number to say, not for us to assume.

[tests/test_transport_parity.py](tests/test_transport_parity.py) checks this: the same sequence of
propose/rollback rounds goes through each of the three transports, and **the tokens must match one for one**. A transport can change the cost but never the result.

### Two HTTP settings that are not optional

Skip these and you end up measuring your own config mistakes instead of the network:

- **`protocol_version = "HTTP/1.1"`**: the stdlib default is 1.0, which closes the connection after every response, so **every step pays a
  TCP handshake**, which adds a wasted RTT and also shows up in `net_excess_ms`.
- **`disable_nagle_algorithm = True`**: otherwise small responses of a few dozen bytes wait for the 40 ms delayed-ACK timer.
  On the client side, `http.client` already sets `TCP_NODELAY`, but the server side doesn't.

There is also a structural one: **one POST carries all of a drafter's requests for that step**. ZMQ gets concurrency by pipelining many messages over one DEALER.
HTTP/1.1 has no usable pipelining, so one POST per request would serialize a drafter's
N sequences into N round trips. Batching per drafter makes the number of round trips depend only on the number of drafters, not on the batch size.
This is asserted with the server-side counter `speculate_posts`, **not** with timing: eight extra round trips on loopback cost under 1 ms,
far too little for timing to catch. It would only show up on a real link.

---

## Verified (real GPUs)

| Stage | What | Result |
|---|---|---|
| **Stage 0** | K1–K7, the seven hook probes, rerun on 0.27.1 | All pass, `runs/stage0_probes.json` |
| **Stage 0** | `custom_class` + `ReplayDraftEngine` end-to-end replay | Output **token-identical** to the greedy baseline |
| **Stage 2a** | Fused: real draft model + `InProcessDraftClient`, Qwen3-1.7B + Qwen3-0.6B, fp32, γ∈{1,2,4,8} | All **token-identical** to the non-speculative baseline |
| **Stage 2b** | Distributed: same two models, draft in a separate process on a separate GPU, real ZMQ socket, RTT sweep | **TPOT linear in RTT, R²=0.98, L̄=2.88** |
| **Stage 2b** | Acceptance length, fused vs ZMQ | Identical, `rel_diff_of_means=0.0` |
| **Stage 3 start** | N-drafter fan-out, 2 real ZMQ drafters + RoundRobin | 5/5 split, tokens identical to the baseline |

The linearity in Stage 2b is the runtime recheck of K2, "`propose()` is on the critical path": it still holds across two real processes and a real
socket.

The scripts are in [scripts/](scripts/) (`verify_*.py`) with matching `slurm_*.sbatch` files.

---

## Drafter state machine and rollback

The slides' "rollback: prefix + verified + bonus, re-prefill" turns into three steps:

1. **Drop last round's speculative tail**: after proposing γ tokens, the drafter is γ positions past the confirmed prefix.
2. **Append the target's authoritative tokens**: these are **not** the tokens it proposed. The last one is the target's
   bonus/correction token, which the drafter has never seen.
3. **Check against the absolute `num_confirmed`**, and on a mismatch **fail loudly** instead of drafting on from a wrong prefix.
   Using an absolute value instead of a delta also makes the protocol idempotent under retries (the HTTP client's retry after a dropped keep-alive connection relies on this).

[`VllmDraftEngine`](dsdreal/draft/engine_vllm.py) uses exactly the approach from the slides: each round it **re-runs `generate()` statelessly**
on the full confirmed prefix, and relies on `enable_prefix_caching` (a public, stable engine flag) to make the repeated prefix cheap,
without touching any vLLM internals. Correctness rests entirely on determinism: with fp32 + `enforce_eager` + greedy,
the same prefix always produces the same continuation, whether or not it hits the prefix cache.

The cost is one extra prefill per round, so this is not a faithful SD rollback. The more "efficient" alternative (a resident, pausable,
rollback-capable session on the draft side) would mean rewriting the target's own KV-rollback scheduling against version-sensitive internals, and a real drafter
would still have to go back to resubmission on every rejection.

---

## Stage 0 constraints

**All probes passing means "plug into vLLM, don't fork" holds.** The full record of the first round on 0.21.0 is in
[docs/stage0-findings.md](docs/stage0-findings.md) (in Chinese). That document covers the original results on WSL / RTX 5070 Ti /
vLLM 0.21.0. The platform has since moved to 0.27.1 / L40S and the probes were rerun, so
`runs/stage0_probes.json` is the one to trust.

Two API drifts hit when upgrading have been fixed in the probes: `NgramProposer.propose()` gained a leading
`num_speculative_tokens` parameter, and `Scheduler.schedule()` gained `throttle_prefills`.

Constraints that still hold:

1. **`draft_probs` is always `None`** (23/23) → acceptance is **token match, lossless at T=0**, not Leviathan-style
   stochastic rejection sampling. The question on the slides, "vLLM V1 sd support greedy only?": on this path, yes.
2. **`RejectionSampler` has a second call site** (`_dummy_sampler_run` during memory profiling, with random logits).
   The L2 tap must have a `live` latch, or the acceptance data gets polluted **without any error** (788 vs 288).
3. **0.27.1 supports `custom_class` natively**, so the `NgramProposer` subclass shim planned in the 0.21.0 days is no longer needed.
   The `GPUModelRunner.__init__` monkeypatch in `plugin.py` has also been removed, since it unconditionally overwrote
   `self.drafter` and broke every non-custom_class run along with it.

---

## Quick start

```bash
source scripts/env.sh          # source this before any entry point

# 1. CPU only: tests for the transport layer, proposer and routing, no GPU or model, runs in seconds
pytest tests -q                # torch is optional (needed only by the EAGLE3 engine tests)

# 2. Needs a GPU
python scripts/verify_vllm_hooks.py            # the seven Stage 0 probes (--probe k1 runs one)
python scripts/verify_replay_proposer.py       # custom_class replay, lossless end to end
python scripts/verify_fused_real_draft.py      # Stage 2a
python scripts/verify_distributed_rtt_sweep.py # Stage 2b (start a draft server first)
sbatch scripts/slurm_fused_real_draft.sbatch   # batch runs on a cluster
```

`requirements/base.txt` deliberately excludes torch/vLLM: the transport layer and its tests **should never need the GPU stack**,
and that file is what guards the boundary. GPU dependencies are in `requirements/gpu.txt` / `pip install -e .[gpu]`.

### Starting a draft server

```bash
# ZMQ (the one Stage 2b uses)
python scripts/run_draft_server.py --drafter-id d0 --bind tcp://0.0.0.0:5560
# HTTP (same engine, different wire)
python scripts/run_draft_server.py --transport http --drafter-id d0 --bind http://0.0.0.0:5560
```

On the target side, hand the client to the proposer **before** constructing the `LLM`:

```python
from dsdreal.proto.transport import make_client
from dsdreal.target import proposer

proposer.configure(make_client("http", addresses={"d0": "http://node1:5560"}))
#                              "zmq",  addresses={"d0": "tcp://node1:5560"}   # swap this line

target = LLM(
    model=..., dtype="float32", enforce_eager=True,
    speculative_config={
        "method": "custom_class",
        "model": "dsdreal.target.proposer.RemoteDraftProposer",
        "num_speculative_tokens": GAMMA,
    },
)
```

`configure()` accepts any `DraftClient`, so switching transports needs no change to the proposer.

---

## Layout

```
dsdreal/
├── proto/      messages.py (msgspec, emits both JSON and msgpack)
│               transport.py (DraftClient + InProcess / Zmq / Http)
├── draft/      base.py  engine_echo.py (no GPU)  engine_vllm.py (real model, batched generate)
│               eagle3_model.py  engine_eagle3.py (remote EAGLE3 head)
│               engine_replay.py (replay)  server.py (ZMQ)  http_server.py (HTTP)
├── target/     proposer.py (custom_class)  hidden_states.py (K8)  plugin.py (entry point)
├── policy/     routing.py (drafter RR, target RR/Random/JSQ)  window.py (SpecDec++)
├── netem/      base.py  null.py  sleep.py (RTT/jitter/latency matrix)
├── router/     target_server.py  frontend.py (N:M)
├── metrics/    records.py  collector.py  export.py
├── workload/   trace.py        experiments/  driver.py
sim/            run dsdSim under the same config: make_config.py  run_sim.py  run_sweep.py
                calibrate_from_run.py  dsdsim_shim.py  vidur/ (A6000 profile patch)
scripts/        stage 0-2 checks: verify_*.py  slurm_*.sbatch
                Table 3 pipeline: run_all_nyulab.sh  table3_sweep.sh  run_nto1.sh  run_nm.sh  make_table3.py
                gates and real-vs-sim: gate_step{A,D}.sh  compare_outputs.py  final_compare.sh  compare_real_sim.py
                (index with one line per script: scripts/README.md)
tests/          CPU-only, no vLLM needed: transport (in-process / ZMQ / HTTP + parity), proposer,
                routing, router, SpecDec++ window, metrics, workload, driver loop, EAGLE3 engine, VIDUR provider
data/           dsdSim traces and acceptance profiles (MIT), VIDUR A6000 profiles
results/        measured Table 3 (table3_final/, table3_client/)
docs/           stage0-findings.md  table3-plan.md  table3-results.md  figures/
docs/notes/     plan.md (full design)  context.md (original brief), both in Chinese
```

## Table 3 (small scale): Vanilla SD / SpecDec++ / EAGLE3 × N drafters

The plan and progress are in [docs/table3-plan.md](docs/table3-plan.md), and **the results (real-system Table 3, gates, real vs sim, n:m) are in [docs/table3-results.md](docs/table3-results.md)** (both in Chinese). The online experiment pipeline:

```
trace (dsdSim gsm8k, Poisson arrivals) ─► experiments/driver.py ─► in-process LLMEngine (target)
                                            │  propose() every step ─► N draft servers (ZMQ)
                                            └► metrics/collector.py ─► dsd_metrics.json / steps.jsonl / actual_run.jsonl
```

| Module | Role |
|---|---|
| [experiments/driver.py](dsdreal/experiments/driver.py) | Calls `add_request` at each arrival time and timestamps tokens step by step (not AsyncLLM: it moves the engine into another process, and the proposer's module-global configuration is lost) |
| [metrics/collector.py](dsdreal/metrics/collector.py) | Derives each round's accepted count from the next round's `num_confirmed` delta, and reports **two conventions** side by side: standard vs dsdSim (no bonus, TTFT to the first round with an accepted token, TPOT aggregated per token) |
| [workload/trace.py](dsdreal/workload/trace.py) | dsdSim trace (`data/dsdsim/`, MIT) → prompts of exact length, and `--chat-template` keeps the total length unchanged |
| [policy/window.py](dsdreal/policy/window.py) | SpecDec++: a line-by-line port of the sim's `SpecPPGammaPolicy`, with per-position acceptance probabilities calibrated on the real system |
| [target/hidden_states.py](dsdreal/target/hidden_states.py) | K8: gives the custom_class proposer the target's EAGLE3 aux hidden states (sliced to the newly confirmed positions) |
| [draft/eagle3_model.py](dsdreal/draft/eagle3_model.py) / [engine_eagle3.py](dsdreal/draft/engine_eagle3.py) | Remote EAGLE3 head (ported from vLLM's llama_eagle3): KV rollback + re-prefill + chained drafting, batched |
| [router/](dsdreal/router/) | N:M: target server processes + a router frontend (RR / Random / JSQ, with TTFT counted from router entry) |
| [sim/](sim/) | Runs dsdSim under the same configuration. Latency comes from one of two sources: a calibration file fitted from real runs by `calibrate_from_run.py` (default), or VIDUR's predictions on A6000 profiles (`--latency vidur`, see [sim/README.md](sim/README.md#latency-from-vidur)) |

Gates (all passed locally on small models, rerun remotely with Llama-3.1-8B / 3.2-1B):

- **Step A**: under online load in fp32, spec output == no-draft baseline (`scripts/compare_outputs.py`), and the accounting `Σ(accepted+1)` matches the output length.
- **Step C**: per-request γ ∈ [2, 8] and equal to the policy's output, still lossless.
- **Step D**: K8 alignment (per-position cos ≥ 0.98 against an HF reference, and clearly higher than off-by-one). Remote EAGLE3 acceptance length is within 5% of vLLM's native `method=eagle3` (locally 1.970 vs 1.973), and output is lossless in fp32 (30/30 locally).

On the remote (NYULAB: free GPUs only, at most 3), one command runs everything: `nohup scripts/run_all_nyulab.sh > runs/pipeline.log 2>&1 &`.
Progress goes to `runs/pipeline.status`, and the results table to `runs/table3/table3.md`.
