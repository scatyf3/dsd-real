#!/bin/bash
# Small-scale Table 3 on NYULAB, <=3 GPUs (context.md): 1 target GPU + 2 draft
# GPUs; N > 2 drafters share draft GPUs (tier T2, see plan.md §7).
#
#   scripts/table3_sweep.sh [OUT] [METHODS] [NS]
#   scripts/table3_sweep.sh runs/table3 "vanilla specpp eagle3" "1 2 4 7"
#
# specpp needs a per-position acceptance profile; it is calibrated first with
# a static gamma=8 run at N=1 (reused if already present).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env_nyulab.sh
OUT=${1:-runs/table3}; METHODS=${2:-vanilla specpp}; NS=${3:-1 2 4 7}
PER_DRAFT_RPS=${PER_DRAFT_RPS:-1.0}; RTT_MS=${RTT_MS:-10}; GAMMA=${GAMMA:-4}
mkdir -p "$OUT"
GPUS=$(scripts/free_gpus.sh 3) || exit 1
IFS=, read -r T D1 D2 <<<"$GPUS"
echo "target GPU $T, draft GPUs $D1,$D2 -> $OUT (per_draft_rps=$PER_DRAFT_RPS rtt=$RTT_MS)"
common() {  # N
    local n=$1 nreq=$(( 120 * $1 )); (( nreq > 400 )) && nreq=400
    echo --n-requests $nreq --per-draft-rps "$PER_DRAFT_RPS" --rtt-ms "$RTT_MS" --chat-template --no-prefix-caching
}

if [[ " $METHODS " == *" specpp "* ]] && [ ! -f "$OUT/calib_g8/profile.json" ]; then
    DSD_DRAFT_KIND=vllm scripts/run_nto1.sh "$OUT/calib_g8" 1 "$T" "$D1" $(common 1) --gamma 8 > "$OUT/calib_g8.log" 2>&1 \
        || { echo "calibration failed"; tail -20 "$OUT/calib_g8.log"; exit 1; }
    $DSD_PY scripts/build_acceptance_profile.py "$OUT/calib_g8" -o "$OUT/calib_g8/profile.json" --edges 64,96
fi

for m in $METHODS; do
    for n in $NS; do
        run="$OUT/${m}_N$n"
        [ -f "$run/dsd_metrics.json" ] && { echo "skip $run (done)"; continue; }
        case $m in
            vanilla) extra=(--gamma "$GAMMA") ;;
            specpp)  extra=(--gamma 8 --gamma-policy specpp --acceptance-profile "$OUT/calib_g8/profile.json") ;;
            eagle3)  extra=(--gamma "$GAMMA" --draft-kind eagle3) ;;
            *) echo "unknown method $m"; exit 1 ;;
        esac
        echo "== $m N=$n"
        kind=vllm; model=${DSD_DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
        [ "$m" = eagle3 ] && kind=eagle3 && model=${EAGLE3_HEAD:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}
        DSD_DRAFT_KIND=$kind DSD_DRAFT_MODEL=$model \
        scripts/run_nto1.sh "$run" "$n" "$T" "$D1,$D2" $(common "$n") "${extra[@]}" > "$run.log" 2>&1 \
            || { echo "$run failed"; tail -20 "$run.log"; }
    done
done
$DSD_PY scripts/make_table3.py "$OUT" | tee "$OUT/table3.md"
