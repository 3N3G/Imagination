#!/usr/bin/env bash
# target_hunt_animals_v2 eval — "policy hunts and eats every visible animal".
# Companion to avoid_animals_v2 (which says the policy ignores animals).
#
# Cells:
#   0: A_full              (concise template, no thinking)
#   1: C_grounded_top2M    (concise template, no thinking)
#   2: B_thinking_top2M    (thinking template, thinking_budget=512)

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0) TAG="predonly";                  TRACK_KEY="predonly_full";          INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    1) TAG="grounded_predonly_top2M";   TRACK_KEY="grounded_predonly_top2M"; INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    2) TAG="think_predonly_top2M";      TRACK_KEY="think_predonly_top2M";    INFER_TEMPLATE="predict_only_thinking_prompt.txt";      THINK_FLAG="--gemini-thinking-budget 512" ;;
    *) echo "ERROR unknown cell ${ID}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_steer_v2"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

echo "=== hunt_animals ID=${ID} TRACK=${TRACK_KEY} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/${INFER_TEMPLATE}" \
    ${THINK_FLAG} \
    --embedding-mode target_hunt_animals_v2 --num-episodes 50 \
    --output-dir "${EVAL_BASE}/target_hunt_animals_v2_50ep" \
    --wandb-name "eval_${TRACK_KEY}_target_hunt_animals_v2_50ep"

echo "=== DONE ID=${ID} ==="
