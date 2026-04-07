#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/antimemorize"
NUM_EPISODES=10

# 4 models × gemini mode only
TAGS=("high_dropout" "low_oracle" "aggressive_reg" "moderate_combo")
TAG=${TAGS[$ID]}
CKPT_DIR="${CKPT_BASE}/antimemorize_${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval antimemorize [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width 1024 \
    --embedding-mode gemini \
    --num-episodes ${NUM_EPISODES} \
    --arch-v2 \
    --wandb-name "antimemorize_${TAG}_gemini" \
    --no-video
