"""vLLM general plugin: install dsd-real's hooks inside every engine process.

vLLM loads `vllm.general_plugins` entry points in process0, the engine-core
process, and every worker process. That is the only place code can run early
enough (and in the right process) to swap the drafter that vLLM constructs
by name in GPUModelRunner.__init__.

Nothing here calls propose(); vLLM does that itself, once per engine step,
at gpu_model_runner.py:4605.
"""
from __future__ import annotations

import json
import os

_installed = False


def _enabled() -> bool:
    return os.environ.get("DSD_TARGET_ENABLED", "0") == "1"


def install_remote_drafter() -> None:
    """Swap GPUModelRunner.drafter for the ZMQ-backed proposer."""
    global _installed
    if _installed:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from dsdreal.target.proposer import RemoteDraftProposer

    original = GPUModelRunner.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if getattr(self, "drafter", None) is not None:
            self.drafter = RemoteDraftProposer(self.vllm_config)

    GPUModelRunner.__init__ = __init__
    _installed = True


def register() -> None:
    """Entry point target. Must stay cheap and import-safe."""
    if _enabled():
        install_remote_drafter()


def register_probe() -> None:
    """K5: record which processes actually loaded this plugin."""
    marker = os.environ.get("DSD_PLUGIN_MARKER")
    if not marker:
        return
    os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
    with open(marker, "a") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "ppid": os.getppid()}) + "\n")
