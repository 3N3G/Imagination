#!/usr/bin/env bash
# Side experiment: GEPA-style prompt-iteration loop for pure Gemini-plays.
# Each iteration: 5 eps with current prompt → Gemini Pro proposes refined
# prompt given trajectories. ~10 iters total.
#
# Cost: ~10 iters × 5 eps × ~$0.5/ep + 10 proposals × ~$0.5 ≈ $30 total.
# Runtime: each gemini-play episode is ~5-10 min; 10 × 5 × ~7 min ≈ 6h.
# Use 12h walltime for safety.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

START_PROMPT="${START_PROMPT:-/home/geney/Imagination/configs/training/templates/action_select_seed.txt}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="/data/user_data/geney/prompt_iter_runs/run_${RUN_TAG}"
NUM_ITERS="${NUM_ITERS:-10}"
EPS_PER_ITER="${EPS_PER_ITER:-5}"
PLAYER_MODEL="${PLAYER_MODEL:-gemini-3-flash-preview}"
PROPOSER_MODEL="${PROPOSER_MODEL:-gemini-3-pro-preview}"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "prompt_iter" \
    --nogpu \
    --partition cpu \
    --mem 32G \
    --time 12:00:00 \
    "$@" \
    -- python -m llm.prompt_iter \
        --start-prompt "${START_PROMPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-iters "${NUM_ITERS}" \
        --eps-per-iter "${EPS_PER_ITER}" \
        --player-model "${PLAYER_MODEL}" \
        --proposer-model "${PROPOSER_MODEL}"
