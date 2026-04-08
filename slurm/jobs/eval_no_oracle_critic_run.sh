#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/no_oracle_critic"
NUM_EPISODES=10

TAGS=("nocritic_ow05" "nocritic_ow10" "withcritic_ow05")

TAG=${TAGS[$ID]}
CKPT_DIR="${CKPT_BASE}/${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval no-oracle-critic [${ID}]: ${TAG} ==="

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
