#!/usr/bin/env bash
# SCALING_C Phase 3 — Gemini grounded labelling on the new top-4M
# PPO-RNN-derived data.
#
# Same recipe as C_grounded_2M:
#   - Template:          predict_state_only_prompt_concise_grounded.txt
#   - Cadence:           5 steps (matches forecast horizon)
#   - Future offset:     5 (use real obs at t+5 as the "future state")
#   - Model:             gemini-3-flash-preview, thinking_budget=0
#   - Predict-only mode (no future block; uses concise grounded template)
#
# Cost estimate: 4M rows / 5 cadence = ~800k calls, ~$400 at flash-3 rates.
# Rate-limited at ~2000 rpm → ~7 hours wall time.
#
# IMPORTANT: confirm user approval before submitting (Gemini API spend).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Phase 2 wrote to /data/user_data/geney/scaling_c_data/ because group_data
# quota is exhausted; downstream phases follow the same convention.
DATA_BASE="/data/user_data/geney/scaling_c_data"
INPUT_DIR="${DATA_BASE}/filtered_trajectories_psf_v3_pporn_1e8_top4M"
OUTPUT_DIR="${DATA_BASE}/gemini_labels_psf_v3_cadence5_grounded_3flash"
TEMPLATE="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_grounded.txt"

# Existence checks happen INSIDE the SLURM job (login node has no /data mount).
if [ ! -f "${TEMPLATE}" ]; then
    echo "ERROR: template not found at ${TEMPLATE}" >&2
    exit 1
fi

echo "=== SCALING_C Phase 3 — Gemini grounded label, cadence=5 ==="
echo "  Input:    ${INPUT_DIR}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Template: ${TEMPLATE}"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "scaling_c_phase3_gemini" \
    --nogpu \
    --partition cpu \
    --mem 32G \
    --time 12:00:00 \
    "$@" \
    -- python -m pipeline.gemini_label \
        --filtered-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --gemini-model gemini-3-flash-preview \
        --thinking-budget 0 \
        --predict-only \
        --template-path "${TEMPLATE}" \
        --future-offset 5
