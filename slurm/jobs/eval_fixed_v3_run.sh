#!/usr/bin/env bash
set -euo pipefail
ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results/v3_ablation"
NUM_EPISODES=10

TAGS=("all_fixes" "no_clip" "no_prescan" "both_ent" "clip_5" "old_behavior")

TAG=${TAGS[$ID]}
CKPT_DIR="${CKPT_BASE}/v3_${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_gemini"

echo "=== Eval v3 ablation [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --layer-width 1024 \
    --embedding-mode gemini \
    --num-episodes ${NUM_EPISODES} \
    --arch-v2 \
    --wandb-name "v3_${TAG}_gemini" \
    --no-video
