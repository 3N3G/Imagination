#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/low_ow"
NUM_EPISODES=10

TAGS=("ow001" "ow005" "ow01" "ow0_control")

TAG=${TAGS[$ID]}
CKPT_DIR="${CKPT_BASE}/low_ow_${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval low_ow [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width 1024 \
    --embedding-mode gemini \
    --num-episodes ${NUM_EPISODES} \
    --arch-v2 \
    --wandb-name "low_ow_${TAG}_gemini" \
    --no-video
