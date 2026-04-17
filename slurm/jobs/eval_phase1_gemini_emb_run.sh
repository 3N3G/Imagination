#!/usr/bin/env bash
# 50-ep online eval matching phase1_gemini_emb_sweep_run.sh (8-way array).
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

case "$ID" in
  0) TAG="beta50"        ;;
  1) TAG="beta100"       ;;
  2) TAG="ofrac0.02"     ;;
  3) TAG="ofrac0.10"     ;;
  4) TAG="ow0.25"        ;;
  5) TAG="ow1.0"         ;;
  6) TAG="freezenone"    ;;
  7) TAG="freezeobspm"   ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

FULL_TAG="phase1_gemini_emb_${TAG}"
CKPT_DIR="${CKPT_BASE}/${FULL_TAG}"
EVAL_DIR="${EVAL_BASE}/${FULL_TAG}"

echo "===================================================================="
echo "[$ID] eval ${FULL_TAG}"
echo "===================================================================="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "gemini_embed" \
    --hidden-dim 3072 \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${FULL_TAG}"
