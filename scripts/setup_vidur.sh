#!/bin/bash
# VIDUR for the dsdSim clone: check it out, add the A6000 SKU and the
# Llama-3.1-8B / Llama-3.2-1B model configs, link our A6000 profiles, install
# the predictor's dependencies into the sim venv.
#
#   scripts/setup_vidur.sh
#
# dsdSim imports VIDUR from <dsdSim>/thirdparty/Sai_speculative_vidur, a private
# fork that is not available to us. Upstream microsoft/vidur has everything
# dsdSim imports (vidur.config, vidur.entities, the random forest predictor),
# so a pinned upstream checkout goes to that path instead.
# VIDUR_SRC=<dir> links an existing tree (e.g. the fork) there instead.
set -euo pipefail
cd "$(dirname "$0")/.."
SCR=${SIM_ROOT:-$(pwd)/.sim}
DSDSIM_ROOT=${DSDSIM_ROOT:-$SCR/dsdSim}
SIMPY=${SIMPY:-$SCR/venv/bin/python}
VIDUR_REPO=${VIDUR_REPO:-https://github.com/microsoft/vidur.git}
VIDUR_REV=${VIDUR_REV:-abae7f63aa857300f5cdc6f5e0d27860cd24721b}
DST=$DSDSIM_ROOT/thirdparty/Sai_speculative_vidur
PATCH=$(pwd)/sim/vidur/a6000_llama3.patch

mkdir -p "$(dirname "$DST")"
if [ -n "${VIDUR_SRC:-}" ]; then
    ln -sfn "$(cd "$VIDUR_SRC" && pwd)" "$DST"
elif [ ! -d "$DST/vidur" ]; then
    git init -q "$DST"
    git -C "$DST" fetch -q --depth 1 "$VIDUR_REPO" "$VIDUR_REV"
    git -C "$DST" checkout -q FETCH_HEAD
fi

if [ -z "${VIDUR_SRC:-}" ]; then
    # our own pinned checkout: pinned commit + the current patch, nothing else
    git -C "$DST" checkout -q -- .
    git -C "$DST" apply "$PATCH"
    echo "applied $(basename "$PATCH")"
elif git -C "$DST" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
elif git -C "$DST" apply --check "$PATCH" 2>/dev/null; then
    git -C "$DST" apply "$PATCH"
    echo "applied $(basename "$PATCH")"
else
    echo "WARNING: $(basename "$PATCH") does not apply to $DST -- add the A6000 SKU and the" >&2
    echo "         Llama-3.1-8B / Llama-3.2-1B model configs by hand (see the patch)" >&2
fi

# our profiles (scripts/profile_vidur_nyulab.sh) live in this repo
for kind in compute cpu_overhead; do
    mkdir -p "$DST/data/profiling/$kind"
    for d in data/vidur/profiling/$kind/*/; do
        [ -d "$d" ] && ln -sfn "$(pwd)/${d%/}" "$DST/data/profiling/$kind/$(basename "$d")"
    done
done
# dsdSim insists that the node's send_recv.csv exists; at tensor_parallel 1 /
# one pipeline stage VIDUR never reads it, so any node's file will do.
[ -e "$DST/data/profiling/network/a6000_single" ] || ln -s a40_pairwise_nvlink "$DST/data/profiling/network/a6000_single"

uv pip install -q --python "$SIMPY" pandas fasteners
DSDSIM_ROOT=$DSDSIM_ROOT "$SIMPY" - <<'EOF'
import sys
sys.path.insert(0, "sim")
import dsdsim_shim
sim = dsdsim_shim.load_sim()
from performance import vidur_realtime
if vidur_realtime.throw_on_use is not None:  # dsdSim swallows the import error until first use
    raise SystemExit(f"VIDUR does not import in the sim venv: {vidur_realtime.throw_on_use!r}")
print("VIDUR at", vidur_realtime._VIDUR_ROOT, "- devices:", sorted(vidur_realtime._DEVICE_ALIAS))
EOF
