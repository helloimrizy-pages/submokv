#!/usr/bin/env bash
# Run Diagnostics 0, 1, and 2 across one or more accelerators.
#
# Every stage of this project has more independent units of work than it has
# accelerators, so the work is split by shard rather than the model by device.
# Each worker holds its own copy of the model and writes its own memoization
# file and its own record. The merge step rebuilds the whole experiment, and it
# refuses shards that disagree on the reference perplexity.
set -euo pipefail

GPUS="${GPUS:-2}"
SEQUENCES="${SEQUENCES:-16}"
BUDGET="${BUDGET:-0.35}"
SAMPLES="${SAMPLES:-30}"
STORE="${STORE:-memory}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_sharded () {
    local command="$1"; shift
    echo "== $command across $GPUS worker(s) =="
    local pids=()
    for ((i = 0; i < GPUS; i++)); do
        .venv/bin/python -m submokv.cli --config configs/olmoe.yaml "$command" \
            --device "cuda:$i" --master-store "$STORE" --sequences "$SEQUENCES" \
            --shard "$i" --num-shards "$GPUS" "$@" \
            > "logs/${command}_s${i}.log" 2>&1 &
        pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ $failed -ne 0 ]]; then
        echo "a worker failed; see logs/${command}_s*.log"; exit 1
    fi
    if [[ $GPUS -gt 1 ]]; then
        .venv/bin/python -m submokv.cli merge --name "${command//-/_}"
    fi
}

mkdir -p logs results
run_sharded diagnostic-0 --expert-layers 0,8 --expert-sample 8
run_sharded diagnostic-1 --budget "$BUDGET" --samples "$SAMPLES"

# Diagnostic 2 is a handful of evaluations, so it runs on one worker.
echo "== diagnostic-2 on one worker =="
.venv/bin/python -m submokv.cli --config configs/olmoe.yaml diagnostic-2 \
    --device cuda:0 --master-store "$STORE" --sequences "$SEQUENCES" --budget "$BUDGET"

echo "ALL_DIAGNOSTICS_DONE"
