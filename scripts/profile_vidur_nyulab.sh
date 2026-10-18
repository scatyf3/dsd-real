#!/bin/bash
# VIDUR profiles of the target and draft models on one free A6000.
#
#   scripts/profile_vidur_nyulab.sh [OUT=runs/vidur_profile]
#   STEPS=cpu_overhead scripts/profile_vidur_nyulab.sh     # any of: mlp attention cpu_overhead
#
# VIDUR's own profilers (vidur/profiling/{mlp,attention}) time the operators of
# sarathi-serve, a 2024 vLLM fork pinned to torch 2.3 / CUDA 12.1, so they get
# their own venv under $BASE, next to the vLLM one. No weights are needed: the
# profilers run dummy weights of the shape sim/vidur/a6000_llama3.patch gives.
# cpu_overhead runs sarathi's whole engine (dummy weights) per batch size and
# times its CPU-side phases: schedule, input prep, sampler (which includes the
# LM head matmul), output processing, and the rest of the step as ray_comm_time.
# Result: $OUT/compute/a6000/<model>/{mlp,attention}.csv and
# $OUT/cpu_overhead/a6000_single/<model>/cpu_overheads.csv, to be fetched into
# data/vidur/profiling/ of the repo:
#
#   ssh NYULAB "cd /mnt/hdd/external/dsd/dsd-real/runs/vidur_profile && tar czf - compute cpu_overhead gpu_check*.txt" \
#       | tar xzf - -C data/vidur/profiling
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
OUT=${1:-runs/vidur_profile}
BASE=${VIDUR_PROF_BASE:-/mnt/hdd/external/dsd/vidur-prof}
MODELS=${MODELS:-"meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-1B-Instruct"}
STEPS=${STEPS:-"mlp attention cpu_overhead"}
VIDUR_REV=${VIDUR_REV:-abae7f63aa857300f5cdc6f5e0d27860cd24721b}      # = scripts/setup_vidur.sh
SARATHI_REV=${SARATHI_REV:-c94dbf438711a4670dbf7477175062e87748999d}  # branch `vidur`
FLASHINFER_WHL=${FLASHINFER_WHL:-https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.0/flashinfer-0.1.0+cu121torch2.3-cp310-cp310-linux_x86_64.whl}  # current at SARATHI_REV
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/hdd/external/dsd/uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/mnt/hdd/external/dsd/uv-python}
export RAY_TMPDIR=${RAY_TMPDIR:-/mnt/hdd/external/dsd/ray-tmp}   # not /tmp: the root disk is nearly full
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}   # nvcc for sarathi's three kernels
PY=$BASE/venv/bin/python

checkout() {  # dir url rev
    [ -d "$1/.git" ] && return
    git init -q "$1"
    git -C "$1" fetch -q --depth 1 "$2" "$3"
    git -C "$1" checkout -q FETCH_HEAD
}
if [ ! -f "$BASE/.ready" ]; then
    mkdir -p "$BASE"
    checkout "$BASE/sarathi-serve" https://github.com/microsoft/sarathi-serve.git "$SARATHI_REV"
    checkout "$BASE/vidur" https://github.com/microsoft/vidur.git "$VIDUR_REV"
    uv venv -q --python 3.10 "$BASE/venv"
    # numpy < 2: torch 2.3 is built against numpy 1.x. flashinfer by wheel URL:
    # its index now serves `flashinfer-python` under the old name, which uv
    # rejects, and sarathi needs the begin_forward API that 0.2 dropped anyway.
    PATH="$CUDA_HOME/bin:$PATH" uv pip install --python "$PY" -e "$BASE/sarathi-serve" "numpy<2" \
        "flashinfer @ $FLASHINFER_WHL" \
        --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
    uv pip install --python "$PY" --no-deps -e "$BASE/vidur"
    touch "$BASE/.ready"
fi
# pinned commit + the current patch (it may have grown since the venv was built)
git -C "$BASE/vidur" checkout -q -- .
git -C "$BASE/vidur" apply "$REPO/sim/vidur/a6000_llama3.patch"

mkdir -p "$OUT" "$RAY_TMPDIR"
OUT=$(cd "$OUT" && pwd)
# Shared box: only a GPU nobody holds. The attention profiler also sizes its KV
# cache to 90% of the card, so the card has to be empty anyway.
GPU=$(scripts/free_gpus.sh 1)
bus=$(nvidia-smi -i "$GPU" --query-gpu=pci.bus_id --format=csv,noheader)
others() {
    nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | grep -i "$bus" \
        | while IFS=', ' read -r b pid; do u=$(ps -o user= -p "$pid"); [ "$u" != "$USER" ] && echo "$b $pid $u"; done || true
}
echo "== profiling on GPU $GPU: $STEPS for $MODELS"
cd "$BASE/vidur"
export CUDA_VISIBLE_DEVICES=$GPU PATH="$BASE/venv/bin:$PATH"
seen=""
for step in $STEPS; do
    case $step in
        mlp|attention) args=(--num_gpus 1) ;;
        cpu_overhead) args=(--max_batch_size 64) ;;  # = max_num_seqs of our target
        *) echo "unknown step $step" >&2; exit 1 ;;
    esac
    # shellcheck disable=SC2086
    "$PY" -m vidur.profiling.$step.main --models $MODELS --num_tensor_parallel_workers 1 "${args[@]}" \
        --output_dir "$OUT/raw"
    seen+=$(others)
done
check=$OUT/gpu_check.$(echo "$STEPS" | tr ' ' '_').txt
{ echo "gpu $GPU $(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader) ($STEPS)"
  [ -n "$seen" ] && echo "OTHER PROCESSES DURING PROFILING (timings suspect):" && echo "$seen" || echo "clean"; } \
    > "$check"

for m in $MODELS; do
    for step in $STEPS; do
        src=$(ls -d "$OUT"/raw/$step/*/ | tail -1)$m
        case $step in
            mlp|attention) src=$src/$step.csv; dst=$OUT/compute/a6000/$m/$step.csv ;;
            cpu_overhead) src=$src/cpu_overhead.csv; dst=$OUT/cpu_overhead/a6000_single/$m/cpu_overheads.csv ;;
        esac
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "$dst: $(($(wc -l < "$dst") - 1)) rows"
    done
done
cat "$check"
