#!/usr/bin/env bash
# v2 positive-target + direction-only steering eval array.
# 9 variants × 4 tracks = 36 array tasks.
#
# Variants (mod 9):
#   0: target_collect_stone_v2
#   1: target_descend_v2
#   2: target_eat_cow_v2
#   3: target_drink_water_v2
#   4: target_place_stone_v2
#   5: direction_left_v2
#   6: direction_right_v2
#   7: direction_up_v2
#   8: direction_down_v2
#
# Tracks (div 9):
#   0 = A_full              psf_v2_cadence5_predonly
#   1 = A_top2M             psf_v2_cadence5_predonly_top2M
#   2 = B_thinking_2M       psf_v2_cadence5_think_predonly_top2M (uses thinking template + thinking_budget 512)
#   3 = C_grounded_2M       psf_v2_cadence5_grounded_predonly_top2M

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "target_collect_stone_v2"
    "target_descend_v2"
    "target_eat_cow_v2"
    "target_drink_water_v2"
    "target_place_stone_v2"
    "direction_left_v2"
    "direction_right_v2"
    "direction_up_v2"
    "direction_down_v2"
)
N_VARIANTS=${#VARIANTS[@]}

V_IDX=$(( ID % N_VARIANTS ))
T_IDX=$(( ID / N_VARIANTS ))

MODE="${VARIANTS[$V_IDX]}"

case "${T_IDX}" in
    0) TAG="predonly";                  TRACK_KEY="predonly_full";       INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    1) TAG="predonly_top2M";            TRACK_KEY="predonly_top2M";      INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    2) TAG="think_predonly_top2M";      TRACK_KEY="think_predonly_top2M"; INFER_TEMPLATE="predict_only_thinking_prompt.txt";      THINK_FLAG="--gemini-thinking-budget 512" ;;
    3) TAG="grounded_predonly_top2M";   TRACK_KEY="grounded_predonly_top2M"; INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    *) echo "ERROR unknown track index ${T_IDX}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_steer_v2"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: ${CKPT} not found" >&2; exit 1
fi

echo "=== steerability v2 ID=${ID}  TRACK=${TRACK_KEY}  MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/${INFER_TEMPLATE}" \
    ${THINK_FLAG} \
    --embedding-mode "${MODE}" --num-episodes 50 \
    --output-dir "${EVAL_BASE}/${MODE}_50ep" \
    --wandb-name "eval_${TRACK_KEY}_${MODE}_50ep"

echo "=== DONE ID=${ID}  TRACK=${TRACK_KEY}  MODE=${MODE} ==="
