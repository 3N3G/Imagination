#!/usr/bin/env bash
# Track B (thinking) OR Track C (grounded_v6) label array — both on the top-2M
# bitpacked subset. Array index selects the variant.
#   0: thinking  (thinking_budget=512, max_output_tokens=1024)
#   1: grounded  (future_offset=5)
# Each variant processes the same 20 bitpacked shards (top-2M is ~20 files worth).
set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID unset" >&2; exit 1
fi
ID=${SLURM_ARRAY_TASK_ID}

INPUT_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/filtered_trajectories_psf_v2_top2M"
OUT_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"

case "${ID}" in
    0)
        TAG="think"
        OUT_DIR="${OUT_BASE}/gemini_labels_psf_v2_cadence5_${TAG}_3flash"
        TEMPLATE="configs/training/templates/predict_only_thinking_prompt.txt"
        EXTRA_ARGS=(--thinking-budget 512)
        echo "=== [${ID}] thinking label (budget=512) on top2M ==="
        ;;
    1)
        TAG="grounded"
        OUT_DIR="${OUT_BASE}/gemini_labels_psf_v2_cadence5_${TAG}_3flash"
        TEMPLATE="configs/training/templates/predict_state_only_prompt_concise_grounded.txt"
        EXTRA_ARGS=(--thinking-budget 0 --future-offset 5)
        echo "=== [${ID}] grounded label (future_offset=5) on top2M ==="
        ;;
    *)
        echo "ERROR: unknown array ID ${ID}" >&2; exit 2
        ;;
esac

mkdir -p "${OUT_DIR}"

python -m pipeline.gemini_label \
    --filtered-dir "${INPUT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --template-path "${TEMPLATE}" \
    --predict-only \
    --gemini-model gemini-3-flash-preview \
    "${EXTRA_ARGS[@]}"

echo "=== DONE label task ${ID} (${TAG}) ==="
