#!/usr/bin/env bash
# Re-eval 3 existing policies so summary.json logs per-step actions
# (needed for analysis/direction_match.py).  Writes to *_dirmatch eval dirs
# so we don't clobber prior runs.
# Array 0=awr_aug_debug, 1=freeze_obs_bcawr, 2=awr_bc_aug_debug.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

TAGS=("awr_aug_debug" "freeze_obs_bcawr" "awr_bc_aug_debug")
TAG=${TAGS[$ID]}

CKPT_DIR="/data/group_data/rl/geney/checkpoints/${TAG}"
EVAL_DIR="/data/group_data/rl/geney/eval_results/${TAG}_dirmatch"

echo "=== Eval [${ID}] ${TAG} → ${EVAL_DIR} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}_dirmatch"

echo ""
echo "=== DONE ${TAG} ==="
