#!/usr/bin/env bash
# Evaluate V2 architecture comparison checkpoints.
# Array 0: eval v2_awr_aug (online, with Gemini+Qwen)
# Array 1: eval v2_awr_bc_aug (online, with Gemini+Qwen)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

TAGS=("v2_awr_aug" "v2_awr_bc_aug")
TAG=${TAGS[$ID]}

CKPT_DIR="${CKPT_BASE}/${TAG}"
EVAL_DIR="${EVAL_BASE}/${TAG}"

echo "=== Eval V2 [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --arch-v2 \
    --num-episodes 50 \
    --eval-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}"
