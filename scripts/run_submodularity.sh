#!/usr/bin/env bash
# Run the Sub-MoKV interaction matrix across one model replica per GPU.
#
# RunPod defaults target a 2x A40 pod. Override any setting as an environment
# variable, for example: SEQUENCES=16 GPUS=2 scripts/run_submodularity.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

GPUS="${GPUS:-2}"
# Unset means "whatever the config pins". These used to default to 64 x 2048,
# which silently moved the experiment off the 4096-token train windows the
# config declares and the noise floor was measured on.
SEQUENCES="${SEQUENCES:-}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-}"
CONFIG="${CONFIG:-configs/olmoe.yaml}"
STORE="${STORE:-checkpoint}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_ARGS=("$@")

export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"

if [[ ! -x .venv/bin/python ]]; then
    echo "missing .venv; run scripts/setup_pod.sh first" >&2
    exit 1
fi
if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS must be a positive integer, got $GPUS" >&2
    exit 1
fi

AVAILABLE_GPUS="$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPUS > AVAILABLE_GPUS )); then
    echo "requested $GPUS GPU workers but torch sees only $AVAILABLE_GPUS" >&2
    exit 1
fi

mkdir -p logs results cache

CALIBRATION_ARGS=()
[[ -n "$SEQUENCES" ]] && CALIBRATION_ARGS+=(--sequences "$SEQUENCES")
[[ -n "$SEQUENCE_LENGTH" ]] && CALIBRATION_ARGS+=(--sequence-length "$SEQUENCE_LENGTH")

echo "== Sub-MoKV diagnostic: $GPUS GPU shard(s), calibration: ${CALIBRATION_ARGS[*]:-from $CONFIG} =="
echo "== config: $CONFIG | master store: $STORE | run: $RUN_ID =="

pids=()
outputs=()
for ((shard = 0; shard < GPUS; shard++)); do
    output="results/submodularity_${RUN_ID}_s${shard}of${GPUS}.json"
    log="logs/submodularity_${RUN_ID}_s${shard}of${GPUS}.log"
    outputs+=("$output")
    echo "starting shard $shard on cuda:$shard -> $log"
    .venv/bin/python scripts/submodularity_diagnostic.py \
        --config "$CONFIG" \
        --device "cuda:$shard" \
        --master-store "$STORE" \
        --local-files-only \
        ${CALIBRATION_ARGS[@]+"${CALIBRATION_ARGS[@]}"} \
        --shard "$shard" \
        --num-shards "$GPUS" \
        --output "$output" \
        "${EXPERIMENT_ARGS[@]}" \
        >"$log" 2>&1 &
    pids+=("$!")
done

failed=0
for ((shard = 0; shard < GPUS; shard++)); do
    if ! wait "${pids[$shard]}"; then
        echo "shard $shard failed; inspect logs/submodularity_${RUN_ID}_s${shard}of${GPUS}.log" >&2
        failed=1
    fi
done
if (( failed != 0 )); then
    exit 1
fi

merged="results/submodularity_${RUN_ID}_merged.json"
.venv/bin/python scripts/submodularity_diagnostic.py \
    --merge-shards "${outputs[@]}" \
    --output "$merged"

echo "SUBMODULARITY_DIAGNOSTIC_DONE"
echo "merged report: $merged"
