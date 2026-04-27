#!/usr/bin/env bash
# SCALING_C Phase 6 — headline eval of the new C_v2 policy
# (psf_v3_pporn_1e8_grounded_top4M/freezenone) across 3 prompts:
#   ID 0: baseline (concise prompt, no special steering)
#   ID 1: achievement_max_v2 (current best on C_grounded_2M = 18.33)
#   ID 2: achievement_max_v2_thresh6 (new threshold-6 + canonical-ordering)
#
# n=30 each, 12h walltime per cell. Output to user_data.

set -euo pipefail

ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "${ID}" ]; then
    echo "Usage: pass cell ID 0|1|2 as first arg or via SLURM array" >&2
    exit 2
fi

CKPT_BASE="/data/user_data/geney/scaling_c_data/checkpoints_psf_v3_pporn_1e8_grounded_top4M/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

EVAL_BASE="/data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_top4M_steer_score"
PROMPT_BASE="/home/geney/Imagination/configs/training/templates"
CONCISE="${PROMPT_BASE}/predict_state_only_prompt_concise.txt"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: missing ${CKPT} — Phase 5 not complete" >&2
    exit 1
fi

case "${ID}" in
    0) MODE="baseline_concise"; ;;
    1) MODE="achievement_max_v2"; ;;
    2) MODE="achievement_max_v2_thresh6"; ;;
    *) echo "unknown cell ID: ${ID}" >&2; exit 2 ;;
esac

OUT_DIR="${EVAL_BASE}/${MODE}_30ep"
mkdir -p "${OUT_DIR}"

echo "=== SCALING_C Phase 6 cell ${ID} (${MODE}) ==="
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
        --wandb-name "eval_psf_v3_pporn_1e8_grounded_top4M_${MODE}_30ep"
else
    python -m eval.eval_online \
        --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
        --embed-backend gemini_embed --hidden-dim 3072 \
        --extract-prediction-only \
        --prompt-template-path "${CONCISE}" \
        --embedding-mode "${MODE}" --num-episodes 30 \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_psf_v3_pporn_1e8_grounded_top4M_${MODE}_30ep"
fi

echo "=== DONE cell ${ID} ==="
