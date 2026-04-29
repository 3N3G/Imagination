#!/usr/bin/env bash
# OOD steering pilot on the ORIGINAL Track C (C_grounded_2M) for comparison.
# Track C was the steerability winner on the specificity matrix (12 WIN / 9 NULL
# / 1 WRONG-WAY); xxhighb might have traded steerability for raw return.
#
# Same prompts as scaling_c_v_ood_eval.sh, but pointed at track C's checkpoint.
# Reads MODE env var (one of explore_ood_v1, v2_eat_bat, v2_enter_mines,
# achievement_max_v2, baseline_concise).

set -euo pipefail

NUM_EPISODES="${NUM_EPISODES:-15}"
MODE="${MODE:-v2_enter_mines}"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_grounded_predonly_top2M/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
EVAL_BASE="/data/user_data/geney/eval_results_temp/track_c_ood_compare"
PROMPT="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt"

if [ ! -f "${CKPT}" ]; then echo "ERROR: missing ${CKPT}" >&2; exit 1; fi

OUT_DIR="${EVAL_BASE}/${MODE}_${NUM_EPISODES}ep"
mkdir -p "${OUT_DIR}"

echo "=== Track C OOD pilot — mode=${MODE}, n=${NUM_EPISODES} ==="
echo "  CKPT: ${CKPT}"
echo "  OUT:  ${OUT_DIR}"

if [ "${MODE}" = "baseline_concise" ]; then
    python -m eval.eval_online \
        --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
        --embed-backend gemini_embed --hidden-dim 3072 \
        --extract-prediction-only \
        --prompt-template-path "${PROMPT}" \
        --num-episodes "${NUM_EPISODES}" \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_track_c_${MODE}_${NUM_EPISODES}ep"
else
    python -m eval.eval_online \
        --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
        --embed-backend gemini_embed --hidden-dim 3072 \
        --extract-prediction-only \
        --prompt-template-path "${PROMPT}" \
        --embedding-mode "${MODE}" --num-episodes "${NUM_EPISODES}" \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_track_c_${MODE}_${NUM_EPISODES}ep"
fi

echo "=== DONE ==="
