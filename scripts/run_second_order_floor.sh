#!/usr/bin/env bash
# Measure the second-order noise floor, which is what epsilon is set from.
#
# The quantity the submodularity test classifies is the second-order difference
#
#     D = [F(S_A + j) - F(S_A)] - [F(S_B + j) - F(S_B)]
#
# so the floor D must clear is the spread of D itself across calibration draws,
# not the spread of one perplexity and not the spread of one delta. One square
# per modality is evaluated, all four corners on the same subsample, repeated
# over non-overlapping subsamples.
#
# This runs on one worker: the whole measurement is twelve allocation states,
# and evaluation is plan-major so installing a state is paid for once.
#
# The floor and the run it calibrates must read the same windows. SEQUENCES is
# unset by default so both take the count from configs/olmoe.yaml.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DEVICE="${DEVICE:-cuda:0}"
SUBSAMPLES="${SUBSAMPLES:-6}"
LAYERS="${LAYERS:-8,4}"
STORE="${STORE:-memory}"
CONFIG="${CONFIG:-configs/olmoe.yaml}"
SEQUENCES="${SEQUENCES:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

export PYTHONPATH=.
# cuBLAS needs this before torch is imported; it cannot be set afterwards.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"

if [[ ! -x .venv/bin/python ]]; then
    echo "missing .venv; run scripts/setup_pod.sh first" >&2
    exit 1
fi

EXTRA=()
[[ -n "$SEQUENCES" ]] && EXTRA+=(--sequences "$SEQUENCES")

mkdir -p logs results
log="logs/second_order_floor_${RUN_ID}.log"
echo "== second-order noise floor on $DEVICE | subsamples $SUBSAMPLES | layers $LAYERS =="
echo "== calibration: ${EXTRA[*]:-from $CONFIG} | log: $log =="

.venv/bin/python -u -m submokv.cli --config "$CONFIG" second-order-floor \
    --device "$DEVICE" \
    --master-store "$STORE" \
    --subsamples "$SUBSAMPLES" \
    --layers "$LAYERS" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    "$@" 2>&1 | tee "$log"

echo "SECOND_ORDER_FLOOR_DONE"
echo "The record is the newest results/second_order_floor__*.json."
echo "Paste its epsilon block into configs/olmoe.yaml before running the matrix."
