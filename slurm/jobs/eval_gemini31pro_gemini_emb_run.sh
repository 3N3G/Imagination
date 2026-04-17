#!/usr/bin/env bash
# Text-generator swap eval: use gemini-3.1-pro-preview (vs default 2.5-flash)
# as the Gemini generator at inference for the 2 top gemini_emb policies.
# Tests hypothesis: does a stronger text generator lift online returns given
# that gemini_emb already reads content in the correct direction? Note this
# IS a training/eval mismatch (training used 3.1-flash-lite texts), so a
# positive lift indicates information-bottleneck, a negative/flat result may
# indicate OOD — either way, a cheap signal before a full retraining sweep.
#
# Array index -> policy:
#   0: psf_freeze_obs_bcawr_gemini_emb_beta30.0   (Exp 18 winner, 15.68)
#   1: phase1_gemini_emb_freezenone               (Phase 1 winner, 16.20)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"

case "$ID" in
  0) TAG="psf_freeze_obs_bcawr_gemini_emb_beta30.0" ;;
  1) TAG="phase1_gemini_emb_freezenone" ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

CKPT_DIR="${CKPT_BASE}/${TAG}"
EVAL_DIR="${EVAL_BASE}/${TAG}_gen31pro"

echo "===================================================================="
echo "[$ID] eval ${TAG}  gen=gemini-3.1-pro-preview  backend=gemini_embed"
echo "===================================================================="

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "gemini_embed" \
    --hidden-dim 3072 \
    --gemini-model "gemini-3.1-pro-preview" \
    --num-episodes 25 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}_gen31pro"
