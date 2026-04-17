#!/usr/bin/env bash
# Hyperparameter sweep on psf_freeze_obs_bcawr for qwen3emb and gemini_emb.
# The two dedicated-embedding-model policies show different content regimes:
#   - qwen3emb reads content but content HURTS (+2pp under adv prompts)
#   - gemini_emb reads mildly helpful content (-1.3pp under adv), but obs-branch weak
#
# Array index -> (encoder, variant):
#   0: qwen3emb × β=1
#   1: qwen3emb × β=3
#   2: qwen3emb × β=30
#   3: qwen3emb × 100K steps (β=10)
#   4: gemini_emb × β=1
#   5: gemini_emb × β=3
#   6: gemini_emb × β=30
#   7: gemini_emb × 100K steps (β=10)
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

case "$ID" in
  0) ENCODER=qwen3emb   BETA=1.0  STEPS=50000  SUFFIX="_beta1.0"   ;;
  1) ENCODER=qwen3emb   BETA=3.0  STEPS=50000  SUFFIX="_beta3.0"   ;;
  2) ENCODER=qwen3emb   BETA=30.0 STEPS=50000  SUFFIX="_beta30.0"  ;;
  3) ENCODER=qwen3emb   BETA=10.0 STEPS=100000 SUFFIX="_long"      ;;
  4) ENCODER=gemini_emb BETA=1.0  STEPS=50000  SUFFIX="_beta1.0"   ;;
  5) ENCODER=gemini_emb BETA=3.0  STEPS=50000  SUFFIX="_beta3.0"   ;;
  6) ENCODER=gemini_emb BETA=30.0 STEPS=50000  SUFFIX="_beta30.0"  ;;
  7) ENCODER=gemini_emb BETA=10.0 STEPS=100000 SUFFIX="_long"      ;;
  *) echo "Unknown array index: $ID" >&2; exit 1 ;;
esac

if [ "${ENCODER}" = "qwen3emb" ]; then
  DATA_DIR="${DATA_BASE}/final_trajectories_psf_qwen3emb"
  ORACLE_DATA="${ORACLE_BASE}/predict_only_final_qwen3emb/trajectories_000000.npz"
  PRETRAINED="${CKPT_BASE}/awr_psf_qwen3emb/final.pth"
  HIDDEN_DIM=4096
else
  DATA_DIR="${DATA_BASE}/final_trajectories_psf_gemini_emb"
  ORACLE_DATA="${ORACLE_BASE}/predict_only_final_gemini_emb/trajectories_000000.npz"
  PRETRAINED="${CKPT_BASE}/awr_psf_gemini_emb/final.pth"
  HIDDEN_DIM=3072
fi

TAG="psf_freeze_obs_bcawr_${ENCODER}${SUFFIX}"
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "===================================================================="
echo "[$ID] ${TAG}"
echo "  Encoder: ${ENCODER}  hidden_dim=${HIDDEN_DIM}"
echo "  Beta:    ${BETA}"
echo "  Steps:   ${STEPS}"
echo "  Data:    ${DATA_DIR}"
echo "  Pretrained: ${PRETRAINED}"
echo "===================================================================="

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 2500 \
    --pretrained-checkpoint "${PRETRAINED}" \
    --freeze-mode obs_branch \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim "${HIDDEN_DIM}" \
    --lr 1e-4 \
    --awr-beta ${BETA} \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps ${STEPS} \
    --save-freq 10000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --wandb-name "${TAG}" \
    --max-dataset-gb 60

echo ""
echo "=== DONE ${TAG} ==="
