#!/usr/bin/env bash
# Task B Step 2: the two floor checks that run before any Task C cell exists.
#
# Both vary one thing at a time against the measured floor, which was taken at
# L8 (W:3->4) | L4 (W:3->4) on subsamples 0-5.
#
#   1. CONDITIONING, binding.  L8 (W:3->4) | L4 (W:3->16).  Same layers, matrix
#      conditioning. Isolates the conditioning move. DECISION.md Amendment 2 C
#      fixed the rule before the number existed: sigma2 inside the baseline
#      square's 95% interval leaves the floor standing, outside it Task B is
#      not finished.
#
#   2. LAYER, context only.  L14 (W:3->4) | L10 (W:3->16).  Matrix conditioning
#      at a Task C layer pair, compared against check 1. Isolates the layer.
#      No rule was declared for this in advance, so it selects no verdict and
#      no branch turns on it. It is measured because the floor was taken at
#      layer 8 while Task C's cells sit at layers 2, 6, 10 and 14, whose gains
#      span 2.3x, and a reader is owed the size of that assumption.
#
# Together: baseline -> check 1 isolates conditioning, check 1 -> check 2
# isolates layer. Roughly 36 evaluations; the baseline's cached states cover
# half of check 1.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BASELINE="${BASELINE:-results/second_order_floor__20260821T200032Z__601002ea.json}"
DEVICE="${DEVICE:-cuda:0}"
SUBSAMPLES="${SUBSAMPLES:-6}"
STORE="${STORE:-checkpoint}"
CONFIG="${CONFIG:-configs/olmoe.yaml}"

export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"

if [[ ! -x .venv/bin/python ]]; then
    echo "missing .venv; run scripts/setup_pod.sh first" >&2
    exit 1
fi
if [[ ! -f "$BASELINE" ]]; then
    echo "no baseline floor record at $BASELINE" >&2
    exit 1
fi

mkdir -p logs results

newest () { ls -t results/second_order_floor_"$1"_check__*.json 2>/dev/null | head -1; }

run_check () {
    local label="$1" layers="$2" mode="$3" against="$4"
    echo
    echo "== Task B check: $label | layers $layers | against $(basename "$against") =="
    .venv/bin/python -u -m submokv.cli --config "$CONFIG" second-order-floor \
        --device "$DEVICE" --master-store "$STORE" \
        --subsamples "$SUBSAMPLES" --layers "$layers" \
        --weight-move 3:4 --weight-conditioning 3:16 \
        --modalities weight_to_weight \
        --compare-to "$against" --comparison "$mode" --check-label "$label" \
        2>&1 | tee "logs/task_b_${label}_check.log"
}

# 1. Conditioning, binding. Compared against the adjacent-conditioning floor.
run_check conditioning "8,4" binding "$BASELINE"
CHECK1="$(newest conditioning)"
if [[ -z "$CHECK1" ]]; then
    echo "the conditioning check wrote no record; stopping" >&2
    exit 1
fi

# 2. Layer, context only. Compared against check 1, so only the layer differs.
run_check layer "14,10" context "$CHECK1"

echo
echo "TASK_B_CHECKS_DONE"
echo "  conditioning (binding): $CHECK1"
echo "  layer (context only):   $(newest layer)"
echo
echo "The conditioning verdict decides whether Task C runs on the measured epsilon."
echo "The layer check is context and selects nothing."
