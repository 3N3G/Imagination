#!/usr/bin/env bash
# Multistep counterfactual direction-match eval (A/B/C at several timesteps).
# Array 0=awr_aug_debug, 1=freeze_obs_bcawr, 2=awr_bc_aug_debug.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

TAGS=("awr_aug_debug" "freeze_obs_bcawr" "awr_bc_aug_debug")
TAG=${TAGS[$ID]}

CKPT_DIR="/data/group_data/rl/geney/checkpoints/${TAG}"
OUT_DIR="/data/group_data/rl/geney/eval_results/${TAG}_dirmatch_cf_multistep"

echo "=== Multistep counterfactual dirmatch [${ID}] ${TAG} → ${OUT_DIR} ==="

python -m eval.eval_direction_counterfactual_multistep \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --output-dir "${OUT_DIR}" \
    --num-episodes 30 \
    --intervention-steps "0,75,150,300" \
    --layer-width 512

echo ""
echo "=== DONE ${TAG} ==="
