#!/usr/bin/env bash
# Parameterized SCALING_C variant evaluator. Reads env vars:
#   VARIANT_TAG     — must match the train script's tag (CKPT_BASE = checkpoints/.../psf_v3_pporn_1e8_grounded_${VARIANT_TAG}/freezenone)
#   ID              — array index 0|1|2
#     0: baseline_concise
#     1: achievement_max_v2
#     2: achievement_max_v2_thresh6

set -euo pipefail

if [ -z "${VARIANT_TAG:-}" ]; then echo "ERROR: VARIANT_TAG required" >&2; exit 2; fi
ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "${ID}" ]; then echo "ERROR: pass ID 0|1|2" >&2; exit 2; fi

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

EVAL_BASE="/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_steer_score"
PROMPT_BASE="/home/geney/Imagination/configs/training/templates"
CONCISE="${PROMPT_BASE}/predict_state_only_prompt_concise.txt"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: missing ${CKPT}" >&2
    exit 1
fi

case "${ID}" in
    0) MODE="baseline_concise" ;;
    1) MODE="achievement_max_v2" ;;
    2) MODE="achievement_max_v2_thresh6" ;;
    *) echo "unknown ID: ${ID}" >&2; exit 2 ;;
esac

OUT_DIR="${EVAL_BASE}/${MODE}_30ep"
mkdir -p "${OUT_DIR}"

echo "=== Variant ${VARIANT_TAG} eval cell ${ID} (${MODE}) ==="
echo "  CKPT: ${CKPT}"
echo "  OUT:  ${OUT_DIR}"

if [ "${MODE}" = "baseline_concise" ]; then
    python -m eval.eval_online \
        --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
        --embed-backend gemini_embed --hidden-dim 3072 \
        --extract-prediction-only \
        --prompt-template-path "${CONCISE}" \
        --num-episodes 30 \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_${MODE}_30ep"
else
    python -m eval.eval_online \
        --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
        --embed-backend gemini_embed --hidden-dim 3072 \
        --extract-prediction-only \
        --prompt-template-path "${CONCISE}" \
        --embedding-mode "${MODE}" --num-episodes 30 \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_${MODE}_30ep"
fi

echo "=== DONE cell ${ID} ==="
