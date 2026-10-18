#!/usr/bin/env python
"""Plain target step time on this GPU: prefill and decode steps of the target
engine alone (no drafter, no speculation) over batch size x prompt length.

    source scripts/env_nyulab.sh
    CUDA_VISIBLE_DEVICES=$(scripts/free_gpus.sh 1) $DSD_PY scripts/bench_target_forward.py \\
        --out runs/bench_target/bench.json

The serving runs only give the target's time inside speculative steps (verify,
with a prefill mixed in now and then) and only at contexts of 52-310 tokens.
This is the controlled counterpart: the same engine as the Table 3 target
(build_target: eager, bf16, prefix caching off), B identical-length random
prompts admitted together, so that

- step 1 is one prefill of B x prompt_len tokens (kept only if every request
  got its first token from it: vLLM splits a prefill over its token budget),
- the following steps are decode steps of B sequences, one token each.

Step time is the wall time of engine.step() up to the end of its GPU work, the
quantity the simulator has to reproduce. scripts/vidur_component_check.py --bench sets VIDUR against it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsdreal.experiments.driver import add_target_args, build_target, now_ms, setup_env, warmup  # noqa: E402


def run_cell(target, tag: str, b: int, prompt_len: int, n_decode: int, rng: random.Random):
    """-> (prefill_ms or None, [decode step ms])."""
    import torch
    engine = target.engine
    params = target.make_params(SimpleNamespace(max_tokens=n_decode + 1))
    for i in range(b):
        engine.add_request(f"{tag}-{i}", {"prompt_token_ids": [rng.randrange(1000, 30000) for _ in range(prompt_len)]},
                           params)
    emitted: dict[str, int] = {}
    dts, seen = [], []  # per step: its wall time, tokens visible per request after it
    while engine.has_unfinished_requests():
        t = now_ms()
        outs = engine.step()
        # step() returns once the forward is queued on the GPU and only the
        # next step reads its tokens: without this a prefill shows up as ~8 ms
        # and its real cost lands in the first decode step
        torch.cuda.synchronize()
        dts.append(now_ms() - t)
        for o in outs:
            emitted[o.request_id] = len(o.outputs[0].token_ids)
        seen.append(dict(emitted))
    # vLLM hands a forward's tokens out `lag` steps after the step that ran it
    # (one step in 0.27.1: step 1 returns nothing, the last step only collects)
    lag = next((i for i, s in enumerate(seen) if s), len(seen))
    prefill, decode = None, []
    for k in range(len(dts) - lag):  # forward k ran in step k, its tokens show after step k + lag
        after = seen[k + lag]
        before = seen[k + lag - 1] if k + lag else {}
        if k == 0:
            if len(after) == b and all(n == 1 for n in after.values()):
                prefill = dts[0]  # else vLLM split the prefill over its token budget
        elif len(before) == b and all(after[r] == before[r] + 1 for r in before):
            decode.append(dts[k])  # every sequence advanced one token, none was being prefilled
    return prefill, decode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_target_args(ap)
    ap.add_argument("--batch-sizes", default="1 2 4 8 16 32 64")
    ap.add_argument("--prompt-lens", default="64 128 256 512 1024")
    ap.add_argument("--decode-steps", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.no_prefix_caching = True
    setup_env()
    target = build_target(args, ap.error)
    warmup(target, args.warmup)

    rng = random.Random(args.seed)
    rows = []
    for b in map(int, args.batch_sizes.split()):
        for plen in map(int, args.prompt_lens.split()):
            pre, dec = [], []
            for r in range(args.repeats + 1):
                p, d = run_cell(target, f"b{b}-l{plen}-r{r}", b, plen, args.decode_steps, rng)
                if r == 0:
                    continue  # first pass at a new shape: allocator / kernel warm-up
                pre += [p] if p is not None else []
                dec += d
            for kind, xs, ctx in (("prefill", pre, 0), ("decode", dec, plen + args.decode_steps // 2)):
                if xs:
                    rows.append({"kind": kind, "batch": b, "prompt_len": plen, "context": ctx, "n": len(xs),
                                 "ms_p50": statistics.median(xs), "ms_mean": statistics.fmean(xs),
                                 "ms_min": min(xs), "ms_max": max(xs)})
            print(f"B={b:3d} prompt={plen:5d}  prefill "
                  f"{statistics.median(pre) if pre else float('nan'):8.2f} ms ({len(pre)})  decode "
                  f"{statistics.median(dec) if dec else float('nan'):7.2f} ms ({len(dec)})", flush=True)

    import torch
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {k: v for k, v in vars(args).items()},
        "env": {"host": platform.node(), "gpu": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "rows": rows}, indent=1))
    target.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
