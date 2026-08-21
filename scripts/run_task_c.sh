#!/usr/bin/env bash
# Milestone 3b Step 3, on a 2xA40 pod.
#
# Two independent experiments, one per GPU, run in parallel:
#
#   cuda:0  the interaction matrix on the real ladder. 64 cells over four
#           layers, 105 unique allocation states, 3 calibration draws per cell.
#           TARGET moves are adjacent steps; CONDITIONING moves stay large.
#
#   cuda:1  per-expert Diagnostic 1. A sensitivity-ranked assignment over all
#           1024 experts against 30 random feasible allocations at one budget,
#           which is the comparison MoPEQ / MxMoE / BT-MoE actually claim.
#
# The matrix is NOT sharded. Its states are shared across cells, so splitting it
# over two GPUs would do 39% more evaluations to save 23% of the wall clock.
# Giving the second GPU a different experiment wastes nothing.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

FLOOR="${FLOOR:-results/second_order_floor__20260821T200032Z__601002ea.json}"
SUBSAMPLES="${SUBSAMPLES:-0,1,2}"
BUDGET="${BUDGET:-0.35}"
SAMPLES="${SAMPLES:-30}"
STORE="${STORE:-checkpoint}"
CONFIG="${CONFIG:-configs/olmoe.yaml}"
PER_EXPERT_CONFIG="${PER_EXPERT_CONFIG:-configs/olmoe_per_expert.yaml}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX_DEVICE="${MATRIX_DEVICE:-cuda:0}"
PER_EXPERT_DEVICE="${PER_EXPERT_DEVICE:-cuda:1}"

export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf-cache}"

if [[ ! -x .venv/bin/python ]]; then
    echo "missing .venv; run scripts/setup_pod.sh first" >&2
    exit 1
fi
if [[ ! -f "$FLOOR" ]]; then
    echo "no floor record at $FLOOR; a classifying run must be given one" >&2
    exit 1
fi

mkdir -p logs results
matrix_log="logs/task_c_matrix_${RUN_ID}.log"
expert_log="logs/task_c_per_expert_${RUN_ID}.log"
matrix_out="results/task_c_matrix_${RUN_ID}.json"

echo "== Task C on $MATRIX_DEVICE and $PER_EXPERT_DEVICE | run $RUN_ID =="
echo "== floor: $FLOOR | subsamples: $SUBSAMPLES =="

.venv/bin/python -u scripts/submodularity_diagnostic.py \
    --config "$CONFIG" \
    --device "$MATRIX_DEVICE" \
    --master-store "$STORE" \
    --local-files-only \
    --floor "$FLOOR" \
    --subsamples "$SUBSAMPLES" \
    --output "$matrix_out" \
    >"$matrix_log" 2>&1 &
matrix_pid=$!
echo "matrix   -> $matrix_log"

.venv/bin/python -u -m submokv.cli --config "$PER_EXPERT_CONFIG" diagnostic-1-per-expert \
    --device "$PER_EXPERT_DEVICE" \
    --master-store "$STORE" \
    --budget "$BUDGET" \
    --samples "$SAMPLES" \
    --subsamples "$SUBSAMPLES" \
    >"$expert_log" 2>&1 &
expert_pid=$!
echo "per-expert -> $expert_log"

failed=0
wait "$matrix_pid"     || { echo "the matrix run failed; see $matrix_log" >&2; failed=1; }
wait "$expert_pid"     || { echo "the per-expert run failed; see $expert_log" >&2; failed=1; }

echo
tail -n 90 "$matrix_log" || true
echo
tail -n 30 "$expert_log" || true
echo
if [[ $failed -ne 0 ]]; then
    echo "TASK_C_INCOMPLETE" >&2
    exit 1
fi
echo "TASK_C_DONE"
echo "  matrix:     $matrix_out"
echo "  per-expert: newest results/diagnostic_1_per_expert__*.json"
