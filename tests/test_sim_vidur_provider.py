"""sim/dsdsim_shim.VidurProvider: how simulator steps are turned into VIDUR
queries and where the overhead goes. A fake predictor stands in for VIDUR, so
this needs neither the dsdSim clone nor VIDUR."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import dsdsim_shim  # noqa: E402

TARGET = {"model_name": "target", "device": "a6000"}
DRAFT = {"model_name": "draft", "device": "a6000"}
OVERHEAD = {"target_step_ms": 10.0, "draft_step_ms": 1.5, "draft_call_ms": 6.0}


class FakeVidur:
    """2 ms per forward + 0.1 ms per position; records every query."""

    def __init__(self):
        self.calls = []

    def __call__(self, profile, requests):
        self.calls.append((profile["model_name"], [(r.phase, r.tokens_per_request) for r in requests]))
        return 2.0 + 0.1 * sum(sum(r.tokens_per_request) for r in requests)


def provider(overhead=OVERHEAD, draft_source="vidur"):
    fake = FakeVidur()
    p = dsdsim_shim.VidurProvider(fake)
    calib = {"mode": "lockstep", "rtt_ms": 10.0, "vidur_overhead": overhead, "draft_vidur": DRAFT,
             "draft": {"base_ms": 9.0, "ms_per_token": 3.0, "ms_per_token_per_seq": 0.5}}
    if draft_source:
        calib["draft_source"] = draft_source
    p.register_target(target_id="t0", model="m", hardware="g",
                      metadata={"role": "target", "vidur": TARGET, "dsdreal_calib": calib})
    return p, fake, calib


def test_draft_is_gamma_decode_steps_plus_overhead():
    p, fake, calib = provider()
    # 3 sequences on the drafter, gamma 4: four single-token decode steps of a batch of 3
    assert p.draft_ms(calib, 4, 3, context=100) == pytest.approx(6.0 + 4 * ((2.0 + 0.1 * 3) + 1.5))
    assert fake.calls == [("draft", [("decode", (1, 1, 1))])]


def test_target_step_is_one_vidur_batch_with_overhead_once():
    p, fake, _ = provider()
    reqs = [dsdsim_shim.phase_request("prefill", [60], [60]),
            dsdsim_shim.phase_request("decode", [4, 8], [100, 120])]
    assert p.batch_ms("t0", reqs) == pytest.approx(2.0 + 0.1 * (60 + 4 + 8) + 10.0)
    assert fake.calls == [("target", [("prefill", (60,)), ("decode", (4, 8))])]
    p.batch_ms("t0", reqs)
    assert len(fake.calls) == 1  # memoised on the batch composition


def test_no_overhead_is_raw_vidur():
    p, _, calib = provider(overhead={})
    assert p.batch_ms("t0", [dsdsim_shim.decode_request(2, 100, 4)]) == pytest.approx(2.0 + 0.1 * 8)
    assert p.draft_ms(calib, 4, 1) == pytest.approx(4 * 2.1)


def test_lockstep_tail_uses_the_configured_draft_source():
    jobs = [SimpleNamespace(draft_id=f"d{i % 2}_r{i:04d}", gamma_override=0, token_count=4,
                            job_type="decode", context_len=100) for i in range(3)]
    p, _, calib = provider()
    # d0 holds 2 of the 3 sequences; the step waits for the slowest drafter
    assert dsdsim_shim.lockstep_extra_ms(calib, jobs, 4, p) == pytest.approx(10.0 + p.draft_ms(calib, 4, 2, 100))
    # no draft_source (EAGLE3 head): the fitted draft block, as before
    p, fake, calib = provider(draft_source=None)
    assert dsdsim_shim.lockstep_extra_ms(calib, jobs, 4, p) == pytest.approx(10.0 + 9.0 + 3.0 * 4 + 0.5 * 4 * 1)
    assert fake.calls == []
