#!/usr/bin/env bash
# 50-ep online eval for the 8 emb-sweep checkpoints. No content probes here —
# those can be added on the top performers afterwards.
#
# Array index matches psf_freeze_emb_sweep_run.sh:
#   0-2: qwen3emb β={1,3,30}
#   3:   qwen3emb long (100K, β=10)
#   4-6: gemini_emb β={1,3,30}
#   7:   gemini_emb long (100K, β=10)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

case "$ID" in
  0) ENCODER=qwen3emb   SUFFIX="_beta1.0"   ;;
  1) ENCODER=qwen3emb   SUFFIX="_beta3.0"   ;;
  2) ENCODER=qwen3emb   SUFFIX="_beta30.0"  ;;
  3) ENCODER=qwen3emb   SUFFIX="_long"      ;;
  4) ENCODER=gemini_emb SUFFIX="_beta1.0"   ;;
  5) ENCODER=gemini_emb SUFFIX="_beta3.0"   ;;
  6) ENCODER=gemini_emb SUFFIX="_beta30.0"  ;;
  7) ENCODER=gemini_emb SUFFIX="_long"      ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

if [ "${ENCODER}" = "qwen3emb" ]; then
  BACKEND=qwen3_embed
  HIDDEN_DIM=4096
else
  BACKEND=gemini_embed
  HIDDEN_DIM=3072
fi

TAG="psf_freeze_obs_bcawr_${ENCODER}${SUFFIX}"
CKPT_DIR="${CKPT_BASE}/${TAG}"
EVAL_DIR="${EVAL_BASE}/${TAG}"

echo "===================================================================="
echo "[$ID] eval ${TAG}  backend=${BACKEND}  dim=${HIDDEN_DIM}"
echo "===================================================================="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "${BACKEND}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}"
