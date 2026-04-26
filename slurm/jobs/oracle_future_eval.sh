#!/usr/bin/env bash
# Train/eval mismatch experiment: how big is the gap between
# (a) regular concise prompt (no future visible — what we deploy with)
# (b) grounded prompt with the actual t+5 future obs filled in via
#     env-state save/restore + 5-step policy rollout (closest possible
#     emulation of the training distribution)
#
# Cells:
#   0: baseline (concise prompt, no future) — control, n=30
#   1: oracle-future eval (grounded prompt + rolled-forward future) — n=30
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_oracle_future"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ "$ID" = "0" ]; then
    NAME="baseline_concise"
    TPL="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt"
    EXTRA=""
elif [ "$ID" = "1" ]; then
    NAME="oracle_future"
    TPL="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise_grounded.txt"
    EXTRA="--oracle-future-embed"
fi

echo "=== oracle_future_eval ID=${ID} NAME=${NAME} TPL=${TPL} EXTRA=${EXTRA} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "${TPL}" \
    --embedding-mode gemini --num-episodes 30 \
    ${EXTRA} \
    --output-dir "${EVAL_BASE}/${NAME}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_oracle_future_${NAME}_30ep"
echo "=== DONE ==="
