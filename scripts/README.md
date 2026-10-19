# scripts/

Every entry point, grouped by what it is for. Scripts call each other by path
from the repo root (`cd "$(dirname "$0")/.."`), so they stay in one flat folder.

Environment: `source scripts/env.sh` on the Torch cluster (slurm, L40S), or
`source scripts/env_nyulab.sh` on NYULAB (plain multi-GPU box, A6000). Both
respect an existing `DSD_VENV` (and `HF_HOME` on NYULAB), so you can point
them at your own environment.

## Stage 0: does vLLM's plug-in surface hold?

| Script | What it does |
|---|---|
| `verify_vllm_hooks.py` | The seven Stage 0 kill-criteria probes (`--probe k1` runs one) |
| `dsd_probe_sched.py` | Scheduler plug points for K3 / K4 (`--scheduler-cls`, swappable waiting queue) |
| `verify_aux_hidden_states.py` | K8: the custom_class proposer gets the target's EAGLE3 aux hidden states at the right positions |
| `verify_replay_proposer.py` | custom_class replay drafter, lossless end to end |
| `_smoke_engine.py` | Minimal vLLM load + generate on the current GPU |
| `slurm_stage0_verify.sbatch`, `slurm_replay_verify.sbatch` | The two above as slurm jobs |

## Stage 2–3: real drafter, real network, N drafters

| Script | What it does |
|---|---|
| `verify_fused_real_draft.py` | Stage 2a gate: real draft model in process, fp32 lossless for γ ∈ {1,2,4,8} |
| `run_draft_server.py` | Draft-side process behind ZMQ or HTTP (`--transport`) |
| `verify_distributed_rtt_sweep.py` | Stage 2b gate: TPOT rises linearly with injected RTT |
| `verify_acceptance_length_parity.py` | Acceptance length over ZMQ == in-process, same model |
| `verify_ndrafter_fanout.py` | One target, round-robin over ≥ 2 real drafters, still lossless |
| `slurm_*.sbatch` | Slurm wrappers for the gates above (L40S partition) |

## Table 3 pipeline (NYULAB, ≤ 3 free GPUs)

| Script | What it does |
|---|---|
| `run_all_nyulab.sh` | Unattended: waits for free GPUs, then Step A gate → Table 3 sweep → Step D gate |
| `free_gpus.sh` | Prints N GPU indices nobody is using; never lands on a held GPU |
| `table3_sweep.sh` | Vanilla / SpecDec++ / EAGLE3 × N drafters |
| `run_nto1.sh` | One N:1 run: N drafter servers + one online target driver |
| `run_nm.sh`, `nm_nyulab.sh` | One N:M run behind the router frontend; RR vs JSQ at equal load |
| `build_acceptance_profile.py` | Per-position acceptance from a calibration run, for SpecDec++ |
| `make_table3.py` | Collects `dsd_metrics.json` into the paper's Table 3 layout (both metric conventions) |
| `deployment_report.py` | Per-run GPU placement and where each step's time goes |
| `bench_target_forward.py` | Target-only prefill / decode step time over batch × length |

## Gates

| Script | What it does |
|---|---|
| `gate_stepA.sh` | fp32 spec == no-draft baseline under online load, plus bf16 accounting check |
| `gate_stepD.sh` | Remote EAGLE3 head: acceptance within 5% of vLLM's native eagle3, fp32 lossless |
| `compare_outputs.py` | Token-for-token comparison of two runs' `outputs.jsonl` |
| `inspect_divergence.py` | Is a mismatch a bug or an fp near-tie? One fp32 HF forward at the divergence |

## Real vs sim

| Script | What it does |
|---|---|
| `final_compare.sh` | Fetches NYULAB runs, calibrates dsdSim from them, writes real-vs-sim tables |
| `compare_real_sim.py` | Real and simulated Table 3 side by side, dsdSim's metric convention |
| `setup_vidur.sh` | VIDUR for the dsdSim clone: A6000 SKU, Llama configs, our profiles |
| `profile_vidur_nyulab.sh` | VIDUR profiles of target and draft on one free A6000 |
| `vidur_component_check.py` | Per-step VIDUR accuracy against a real run |
| `pack_results.py` | Packs a `final_compare.sh` output dir into a small tracked `results/<name>/` |
