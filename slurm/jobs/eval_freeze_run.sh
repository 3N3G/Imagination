#!/usr/bin/env bash
# Evaluate all 4 freeze experiment checkpoints with eval_online.py (live Gemini+Qwen).
# Array job: 0-3 matching freeze_bc_run.sh config indices.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

TAGS=("freeze_obs_bc" "freeze_obs_bcawr" "freeze_all_bc" "freeze_all_bcawr")
TAG=${TAGS[$ID]}

CKPT_DIR="${CKPT_BASE}/${TAG}"
EVAL_DIR="${EVAL_BASE}/${TAG}"

echo "=== Eval Freeze [${ID}]: ${TAG} ==="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --num-episodes 50 \
    --eval-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}"
