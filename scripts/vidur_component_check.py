#!/usr/bin/env python
"""Per-step accuracy of VIDUR against a real dsd-real run (the component-level
check under the end-to-end real-vs-sim tables).

    DSDSIM_ROOT=.sim/dsdSim .sim/venv/bin/python scripts/vidur_component_check.py \\
        runs/table3_client_final/{vanilla,specpp}_N{1,2,4,7} \\
        --fit-on runs/table3_client_final/{vanilla,specpp}_N1 --fit-overhead vidur_overhead.json

Every engine step of a run (rows of steps.jsonl sharing t_send_ms = one
propose() call, as in sim/calibrate_from_run.py) is rebuilt as the batch the
simulator would hand VIDUR and its prediction is set against the measured time:

- verify: the target forward that checked a call's drafts (next call's t_send -
  this call's t_recv), as one decode PhaseRequest -- dsdSim's mapping: gamma_i
  positions per sequence at context num_confirmed_i + gamma_i;
- verify+prefill: the same when the forward also prefilled newly admitted
  prompts (one VIDUR batch holding both parts). Steps in which no sequence
  carried over are left out: the interval then includes idle time;
- draft: a drafter's wall time for its share of a call (draft_ms), as gamma
  autoregressive decode steps of the draft model ("steps", what the sim uses)
  and as dsdSim's native single forward of gamma tokens ("native");
- with --bench, the plain prefill and decode steps of the target alone
  (scripts/bench_target_forward.py): VIDUR's home ground, no speculation.

VIDUR prices the GPU operators of the transformer blocks only, so "raw" is
expected to sit below the measured wall time; --cpu-overhead turns on VIDUR's
own CPU-overhead model (sarathi's CPU-side step phases, the sampler with the
LM head among them). --fit-overhead fits what a real
step adds on the --fit-on runs only (default: all runs given) and the "+oh"
columns show VIDUR plus that overhead on every run -- fit on N=1, read the
other N as held out.

Needs scripts/setup_vidur.sh and a profile for --device. Markdown on stdout.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
import dsdsim_shim  # noqa: E402
import make_config  # noqa: E402
from calibrate_from_run import fit, load_calls  # noqa: E402

BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 64))


def steps_of(run: Path):
    """-> (target rows, draft rows) of one run.
    target row: (kind, B, verify [(gamma, ctx)], prefill [prompt_len], measured_ms)
    draft row:  (B on that drafter, gamma, ctx, measured_ms)"""
    calls = load_calls(run)
    prompt = {r["request_id"]: r["prompt_len"]
              for r in json.loads((run / "dsd_metrics.json").read_text())["per_request"]}
    target, draft = [], []
    seen: set[str] = set()
    for i, call in enumerate(calls):
        here = {r["request_id"] for r in call}
        seen |= here
        by_drafter: dict[str, list] = {}
        for r in call:
            by_drafter.setdefault(r["drafter_id"], []).append(r)
        for rows in by_drafter.values():
            draft.append((len(rows), max(r["gamma"] for r in rows),
                          round(sum(r["num_confirmed"] for r in rows) / len(rows)), rows[0]["draft_ms"]))
        if i + 1 == len(calls):
            continue
        nxt = {r["request_id"] for r in calls[i + 1]}
        fresh = nxt - seen
        if fresh and not (nxt & here):
            continue  # nothing carried over: the engine may have sat idle in between
        verify = [(r["n_draft"], r["num_confirmed"] + r["n_draft"]) for r in call]
        target.append(("verify+prefill" if fresh else "verify", len(call), verify,
                       [prompt[r] for r in sorted(fresh)], calls[i + 1][0]["t_send_ms"] - call[0]["t_recv_ms"]))
    return target, draft


def predict_target(p, prof, verify, prefill) -> float:
    reqs = []
    if prefill:
        reqs.append(dsdsim_shim.phase_request("prefill", prefill, prefill))
    reqs.append(dsdsim_shim.phase_request("decode", [g for g, _ in verify], [c for _, c in verify]))
    return p.step_ms(prof, reqs)


def summarize(title: str, rows, with_oh: bool) -> list[str]:
    """rows: (run, B, measured, raw, with overhead)."""
    out = [f"### {title}", "",
           "| run | B | n | measured p50 (ms) | VIDUR raw p50 | raw err p50 | raw MAPE |"
           + (" +oh err p50 | +oh MAPE |" if with_oh else ""),
           "|---|---|---|---|---|---|---|" + ("---|---|" if with_oh else "")]
    for run in list(dict.fromkeys(r[0] for r in rows)) + ["all"]:
        sel = [r for r in rows if run == "all" or r[0] == run]
        for lo, hi in BUCKETS + ((1, 10 ** 9),):
            a = np.array([(m, raw, oh) for _, b, m, raw, oh in sel if lo <= b <= hi and m > 0])
            if not len(a):
                continue
            err, err_oh = (a[:, 1] - a[:, 0]) / a[:, 0], (a[:, 2] - a[:, 0]) / a[:, 0]
            label = "all" if hi > 64 else (str(lo) if lo == hi else f"{lo}-{hi}")
            line = (f"| {run} | {label} | {len(a)} | {np.median(a[:, 0]):.1f} | {np.median(a[:, 1]):.1f} "
                    f"| {np.median(err):+.0%} | {np.abs(err).mean():.0%} |")
            if with_oh:
                line += f" {np.median(err_oh):+.0%} | {np.abs(err_oh).mean():.0%} |"
            out.append(line)
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="+", type=Path)
    ap.add_argument("--device", default="a6000", help="VIDUR device SKU")
    ap.add_argument("--cache", default=os.environ.get("VIDUR_CACHE", str(ROOT / ".sim" / "vidur_cache")),
                    help="where VIDUR keeps its trained predictors")
    ap.add_argument("--cpu-overhead", action="store_true",
                    help="VIDUR with its own CPU-overhead model (needs cpu_overheads.csv profiles)")
    ap.add_argument("--fit-overhead", metavar="JSON", default=None, help="fit the overhead and write it here")
    ap.add_argument("--fit-on", nargs="+", type=Path, default=None, help="runs the overhead is fitted on")
    ap.add_argument("--overhead", metavar="JSON", default=None, help="use this overhead instead of fitting")
    ap.add_argument("--no-draft", action="store_true",
                    help="target steps only (runs whose drafter is not the 1B model, e.g. EAGLE3)")
    ap.add_argument("--bench", metavar="JSON", default=None,
                    help="also check the plain prefill / decode steps of scripts/bench_target_forward.py")
    ap.add_argument("--csv", default=None, help="write every step (measured, predicted) here")
    a = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):  # dsdSim and VIDUR print while loading / training
        dsdsim_shim.load_sim()
        p = dsdsim_shim.VidurProvider(dsdsim_shim._runner_predict(a.cache, a.cpu_overhead))
        tprof = make_config.vidur_profile(make_config.TARGET_MODEL, a.device)
        dprof = make_config.vidur_profile(make_config.DRAFT_MODEL, a.device)
        data = {}
        for run in a.run:
            target, draft = steps_of(run)
            t = [(kind, b, m, predict_target(p, tprof, v, pre)) for kind, b, v, pre, m in target]
            d = [] if a.no_draft else [
                (b, g, m, p.step_ms(dprof, [dsdsim_shim.decode_request(b, c)]),
                 p.step_ms(dprof, [dsdsim_shim.decode_request(b, c, g)])) for b, g, c, m in draft]
            data[run] = (t, d)

    oh = {"target_step_ms": 0.0, "draft_step_ms": 0.0, "draft_call_ms": 0.0}
    with_oh = bool(a.fit_overhead or a.overhead)
    if a.overhead:
        oh.update(json.loads(Path(a.overhead).read_text()))
    elif a.fit_overhead:
        fit_runs = a.fit_on or a.run
        missing = [str(r) for r in fit_runs if r not in data]
        if missing:
            ap.error(f"--fit-on runs must be among the runs given: {missing}")
        # target: what a pure verify step takes beyond VIDUR's GPU time
        res = [m - raw for r in fit_runs for kind, _, m, raw in data[r][0] if kind == "verify"]
        oh["target_step_ms"] = round(float(np.median(res)), 3)
        report = {"fit_on": [str(r) for r in fit_runs], "target": {"n": len(res)}}
        # draft: measured = call + gamma * (VIDUR step + per-step overhead)
        D = np.array([(g, m - g * step) for r in fit_runs for _, g, m, step, _ in data[r][1]])
        if len(D):
            if np.ptp(D[:, 0]) == 0:  # gamma never varied: all of it per step (as calibrate_from_run)
                c = np.array([0.0, float(np.median(D[:, 1] / D[:, 0]))])
                report["draft"] = {"n": len(D), "note": "gamma constant in the fit runs: draft_call_ms fixed at 0"}
            else:
                c, r2, p90 = fit(np.column_stack([np.ones(len(D)), D[:, 0]]), D[:, 1])
                report["draft"] = {"n": len(D), "r2": r2, "abs_resid_p90_ms": p90}
            oh.update(draft_call_ms=round(float(c[0]), 3), draft_step_ms=round(float(c[1]), 3))
        Path(a.fit_overhead).write_text(json.dumps({**oh, "_fit": report}, indent=2, default=float))

    trows, drows, nrows, flat = [], [], [], []
    for run, (t, d) in data.items():
        name = run.name
        for kind, b, m, raw in t:
            trows.append((kind, (name, b, m, raw, raw + oh["target_step_ms"])))
            flat.append((name, kind, b, "", m, raw, raw + oh["target_step_ms"]))
        for b, g, m, step, native in d:
            steps = g * step
            steps_oh = oh["draft_call_ms"] + g * (step + oh["draft_step_ms"])
            drows.append((name, b, m, steps, steps_oh))
            nrows.append((name, b, m, native, native))
            flat.append((name, "draft", b, g, m, steps, steps_oh))

    print(f"VIDUR{' with its CPU-overhead model' if a.cpu_overhead else ''} vs measured per step, device {a.device} "
          f"({make_config.TARGET_MODEL} target, {make_config.DRAFT_MODEL} drafter)\n")
    if with_oh:
        src = a.overhead or f"fitted on {', '.join(r.name for r in (a.fit_on or a.run))}"
        print(f"Overhead ({src}): target step {oh['target_step_ms']} ms, "
              f"draft call {oh['draft_call_ms']} ms + {oh['draft_step_ms']} ms per draft step\n")
    lines = []
    for kind in ("verify", "verify+prefill"):
        sel = [r for k, r in trows if k == kind]
        if sel:
            lines += summarize(f"Target {kind} step", sel, with_oh)
    if drows:
        lines += summarize("Draft round, gamma decode steps (the sim's model)", drows, with_oh)
        lines += summarize("Draft round, dsdSim native mapping (one forward of gamma tokens)", nrows, False)
    print("\n".join(lines))
    if a.bench:
        # no speculation there, so no proposer bookkeeping in a step: the
        # overhead column is only indicative for these rows
        print("### Plain target step (scripts/bench_target_forward.py)\n")
        print("| step | B | prompt len | measured p50 (ms) | VIDUR raw | raw err |" + (" +oh err |" if with_oh else ""))
        print("|---|---|---|---|---|---|" + ("---|" if with_oh else ""))
        for r in json.loads(Path(a.bench).read_text())["rows"]:
            b, plen, m = r["batch"], r["prompt_len"], r["ms_p50"]
            req = (dsdsim_shim.phase_request("prefill", [plen] * b, [plen] * b) if r["kind"] == "prefill"
                   else dsdsim_shim.decode_request(b, r["context"]))
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    raw = p.step_ms(tprof, [req])
            except dsdsim_shim.VidurRangeError:  # e.g. a prefill batch of > 4096 tokens
                print(f"| {r['kind']} | {b} | {plen} | {m:.1f} | out of VIDUR's range | – |" + (" – |" if with_oh else ""))
                continue
            line = f"| {r['kind']} | {b} | {plen} | {m:.1f} | {raw:.1f} | {(raw - m) / m:+.0%} |"
            if with_oh:
                line += f" {(raw + oh['target_step_ms'] - m) / m:+.0%} |"
            print(line)
            flat.append((f"bench:{plen}", r["kind"], b, "", m, raw, raw + oh["target_step_ms"]))
        print()
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "kind", "batch", "gamma", "measured_ms", "vidur_raw_ms", "vidur_overhead_ms"])
            w.writerows(flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
