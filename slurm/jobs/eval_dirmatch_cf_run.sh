#!/usr/bin/env bash
# Step-0 counterfactual direction-match eval (3 conditions per episode).
# Array 0=awr_aug_debug, 1=freeze_obs_bcawr, 2=awr_bc_aug_debug.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

TAGS=("awr_aug_debug" "freeze_obs_bcawr" "awr_bc_aug_debug")
TAG=${TAGS[$ID]}

CKPT_DIR="/data/group_data/rl/geney/checkpoints/${TAG}"
OUT_DIR="/data/group_data/rl/geney/eval_results/${TAG}_dirmatch_cf"

echo "=== Counterfactual dirmatch [${ID}] ${TAG} → ${OUT_DIR} ==="

python -m eval.eval_direction_counterfactual \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --num-episodes 50 \
    --layer-width 512

echo ""
echo "=== DONE ${TAG} ==="
