#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/partition"
NUM_EPISODES=10

TAGS=("part_ow0" "part_ow01" "part_ow05" "part_ow10")

TAG=${TAGS[$ID]}
CKPT_DIR="${CKPT_BASE}/${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval partition [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width 1024 \
    --embedding-mode gemini \
    --num-episodes ${NUM_EPISODES} \
    --arch-v2 \
    --wandb-name "${TAG}_gemini" \
    --no-video
