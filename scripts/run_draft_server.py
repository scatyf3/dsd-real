#!/usr/bin/env python
"""Stage 2b: draft-side process. Runs a real vLLM model behind a ZMQ
ROUTER, so the target reaches it over a real socket instead of an in-process
call -- the whole point of 2b versus 2a.

For N-drafter fan-out, launch one of these per drafter with a distinct
--drafter-id and --bind, then point ZmqDraftClient at all of them keyed by
that same id: {"d0": addr0, "d1": addr1, ...}.

    python scripts/run_draft_server.py --bind tcp://0.0.0.0:5560 \\
        --drafter-id d0 --ready-file runs/draft_ready
"""
from __future__ import annotations

import argparse
import os
import signal
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DSD_DRAFT_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--bind", default="tcp://0.0.0.0:5560")
    ap.add_argument("--drafter-id", default="d0",
                     help="must match the key ZmqDraftClient's address dict "
                          "uses for this process")
    ap.add_argument("--ready-file", default=None,
                     help="written with the bound address once serving starts")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This process has its own dedicated GPU and no module-global config to
    # share with a parent -- unlike the target side, multiprocessing here
    # would be fine, but there is nothing to gain from it for one small model.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    print("draft server: constructing LLM...", flush=True)
    from vllm import LLM

    from dsdreal.draft.engine_vllm import VllmDraftEngine
    from dsdreal.draft.server import DraftServer

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_frac,
              enforce_eager=True, dtype="float32", disable_log_stats=True,
              max_model_len=1024)
    print("draft server: LLM constructed, wiring VllmDraftEngine...", flush=True)
    engine = VllmDraftEngine(llm, node_id=args.drafter_id)
    print(f"draft server: binding ROUTER to {args.bind}...", flush=True)
    server = DraftServer(engine, args.bind)
    print(f"draft server: bound to {server.bound_addr}, starting serve thread...", flush=True)
    server.start()
    print(f"draft server serving on {server.bound_addr}", flush=True)
    if args.ready_file:
        with open(args.ready_file, "w") as fh:
            fh.write(server.bound_addr)

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    while not stop["flag"]:
        signal.pause()

    print(f"draft server stopping; served {engine.calls} requests", flush=True)
    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
