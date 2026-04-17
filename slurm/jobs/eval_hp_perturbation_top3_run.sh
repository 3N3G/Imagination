#!/usr/bin/env bash
# HP/Food perturbation eval for the 3 current top policies.
#
# Array index -> policy:
#   0: qwen3gen           psf_freeze_obs_bcawr
#   1: qwen3emb β=3       psf_freeze_obs_bcawr_qwen3emb_beta3.0
#   2: gemini_emb β=30    psf_freeze_obs_bcawr_gemini_emb_beta30.0
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

case "$ID" in
  0) TAG="psf_freeze_obs_bcawr"                       BACKEND="qwen3_gen"     DIM=4096 ;;
  1) TAG="psf_freeze_obs_bcawr_qwen3emb_beta3.0"      BACKEND="qwen3_embed"   DIM=4096 ;;
  2) TAG="psf_freeze_obs_bcawr_gemini_emb_beta30.0"   BACKEND="gemini_embed"  DIM=3072 ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

CKPT_DIR="${CKPT_BASE}/${TAG}"
OUT_DIR="${EVAL_BASE}/${TAG}_hp_perturb"

echo "===================================================================="
echo "[$ID] hp_perturb ${TAG}  backend=${BACKEND}  dim=${DIM}"
echo "===================================================================="

python -m eval.eval_hp_perturbation \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "${BACKEND}" \
    --hidden-dim "${DIM}" \
    --num-episodes 10 \
    --probe-every 1 \
    --output-dir "${OUT_DIR}"
