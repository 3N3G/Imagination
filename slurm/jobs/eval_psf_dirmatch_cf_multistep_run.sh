#!/usr/bin/env bash
# Multistep obs-vs-emb counterfactual direction match for the 3 PSF
# freeze_obs_bcawr policies. 30 eps × 3 policies × 4 intervention steps
# (0/75/150/300) using the matching embedder per policy.
#
# Array index -> policy:
#   0: qwen3gen
#   1: qwen3emb
#   2: gemini_emb
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

ENCODERS=(qwen3gen qwen3emb gemini_emb)
BACKENDS=(qwen3_gen qwen3_embed gemini_embed)
HIDDEN_DIMS=(4096 4096 3072)

ENCODER=${ENCODERS[$ID]}
BACKEND=${BACKENDS[$ID]}
HIDDEN_DIM=${HIDDEN_DIMS[$ID]}

if [ "${ENCODER}" = "qwen3gen" ]; then
    TAG="psf_freeze_obs_bcawr"
else
    TAG="psf_freeze_obs_bcawr_${ENCODER}"
fi

CKPT_DIR="/data/group_data/rl/geney/checkpoints/${TAG}"
OUT_DIR="/data/group_data/rl/geney/eval_results/${TAG}_dirmatch_cf_multistep"

echo "===================================================================="
echo "[$ID] ${TAG}  backend=${BACKEND}  dim=${HIDDEN_DIM}"
echo "===================================================================="

python -m eval.eval_direction_counterfactual_multistep \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "${BACKEND}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --output-dir "${OUT_DIR}" \
    --num-episodes 30 \
    --intervention-steps "0,75,150,300"
