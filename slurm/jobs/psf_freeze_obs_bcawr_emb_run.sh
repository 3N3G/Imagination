#!/usr/bin/env bash
# freeze_obs_bcawr pipeline on the remaining two encoders, with PSF labels
# throughout.  Matches the Apr-12 Exp-1 best config (freeze obs_branch + AWR+BC,
# OF=0.05, OW=0.5, 50K steps, LR=1e-4) but swaps the encoder for the Qwen text
# embeddings on both training and golden/BC data.
#
# Array configs:
#   0 -> qwen3emb    (Qwen3-Embedding-8B, 4096-d)
#   1 -> gemini_emb  (gemini-embedding-001,  3072-d)
#
# Prerequisite: the corresponding golden PSF dataset must exist.
#   reembed_golden_psf.sh (array 0=qwen3emb, 1=gemini_emb) produces the
#   predict_only_final_<backend> directory used as --oracle-data below.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

case "$ID" in
  0)
    TAG="psf_freeze_obs_bcawr_qwen3emb"
    DATA_DIR="${DATA_BASE}/final_trajectories_psf_qwen3emb"
    ORACLE_DATA="${ORACLE_BASE}/predict_only_final_qwen3emb/trajectories_000000.npz"
    PRETRAINED="${CKPT_BASE}/awr_psf_qwen3emb/final.pth"
    HIDDEN_DIM=4096
    ;;
  1)
    TAG="psf_freeze_obs_bcawr_gemini_emb"
    DATA_DIR="${DATA_BASE}/final_trajectories_psf_gemini_emb"
    ORACLE_DATA="${ORACLE_BASE}/predict_only_final_gemini_emb/trajectories_000000.npz"
    PRETRAINED="${CKPT_BASE}/awr_psf_gemini_emb/final.pth"
    HIDDEN_DIM=3072
    ;;
  *)
    echo "Unknown array index: $ID" >&2
    exit 1
    ;;
esac

SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "===================================================================="
echo "[$ID] ${TAG}"
echo "  Data:       ${DATA_DIR}"
echo "  Oracle:     ${ORACLE_DATA}"
echo "  Pretrained: ${PRETRAINED}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo "  Freeze:     obs_branch"
echo "  AWR+BC:     OF=0.05 OW=0.5"
echo "===================================================================="

if [[ ! -f "${ORACLE_DATA}" ]]; then
    echo "ERROR: oracle data missing: ${ORACLE_DATA}" >&2
    echo "Run reembed_golden_psf.sh first." >&2
    exit 1
fi
if [[ ! -f "${PRETRAINED}" ]]; then
    echo "ERROR: pretrained checkpoint missing: ${PRETRAINED}" >&2
    exit 1
fi

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
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 50000 \
    --save-freq 10000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --wandb-name "${TAG}" \
    --max-dataset-gb 60
