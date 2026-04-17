#!/usr/bin/env bash
# Evaluate the PSF freeze checkpoints: 4-config Apr-12-Exp-1 rerun + 2 encoders.
# Array index:
#   0: psf_freeze_obs_bc               (qwen3gen, BC only)
#   1: psf_freeze_obs_bcawr            (qwen3gen, BC+AWR)   ← main comparison
#   2: psf_freeze_all_bc               (qwen3gen, BC only, all frozen)
#   3: psf_freeze_all_bcawr            (qwen3gen, BC+AWR, all frozen)
#   4: psf_freeze_obs_bcawr_qwen3emb   (qwen3emb BC+AWR)
#   5: psf_freeze_obs_bcawr_gemini_emb (gemini_emb BC+AWR)
#
# Each run: 50-ep live Gemini + matching embedder.
# Validation (held-out PSF training + golden PSF) is run inline beforehand.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

CKPT_BASE="/data/group_data/rl/geney/checkpoints"
EVAL_BASE="/data/group_data/rl/geney/eval_results"
DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"

TAGS=(
    "psf_freeze_obs_bc"
    "psf_freeze_obs_bcawr"
    "psf_freeze_all_bc"
    "psf_freeze_all_bcawr"
    "psf_freeze_obs_bcawr_qwen3emb"
    "psf_freeze_obs_bcawr_gemini_emb"
)
BACKENDS=(
    "qwen3_gen" "qwen3_gen" "qwen3_gen" "qwen3_gen"
    "qwen3_embed" "gemini_embed"
)
TRAIN_DIRS=(
    "${DATA_BASE}/final_trajectories_psf_qwen3gen"
    "${DATA_BASE}/final_trajectories_psf_qwen3gen"
    "${DATA_BASE}/final_trajectories_psf_qwen3gen"
    "${DATA_BASE}/final_trajectories_psf_qwen3gen"
    "${DATA_BASE}/final_trajectories_psf_qwen3emb"
    "${DATA_BASE}/final_trajectories_psf_gemini_emb"
)
ORACLE_PATHS=(
    "${ORACLE_BASE}/predict_only_final/trajectories_000000.npz"
    "${ORACLE_BASE}/predict_only_final/trajectories_000000.npz"
    "${ORACLE_BASE}/predict_only_final/trajectories_000000.npz"
    "${ORACLE_BASE}/predict_only_final/trajectories_000000.npz"
    "${ORACLE_BASE}/predict_only_final_qwen3emb/trajectories_000000.npz"
    "${ORACLE_BASE}/predict_only_final_gemini_emb/trajectories_000000.npz"
)
HIDDEN_DIMS=(4096 4096 4096 4096 4096 3072)

TAG=${TAGS[$ID]}
BACKEND=${BACKENDS[$ID]}
TRAIN_DIR=${TRAIN_DIRS[$ID]}
ORACLE_DATA=${ORACLE_PATHS[$ID]}
HIDDEN_DIM=${HIDDEN_DIMS[$ID]}

CKPT_DIR="${CKPT_BASE}/${TAG}"
EVAL_DIR="${EVAL_BASE}/${TAG}"

echo "===================================================================="
echo "[$ID] eval ${TAG}  backend=${BACKEND}  dim=${HIDDEN_DIM}"
echo "===================================================================="

echo ""
echo "--- Held-out PSF training data (files 117-124, last 8) ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${TRAIN_DIR}" \
    --file-offset 117 --max-files 4 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0

echo ""
echo "--- Golden PSF data (BC distribution) ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "$(dirname "${ORACLE_DATA}")" \
    --file-offset 0 --max-files 1 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0

echo ""
echo "--- 50-ep live eval (backend=${BACKEND}) ---"
EXTRA_ARGS=()
if [ "${BACKEND}" != "qwen3_gen" ]; then
    EXTRA_ARGS+=(--embed-backend "${BACKEND}" --hidden-dim "${HIDDEN_DIM}")
fi

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name "eval_${TAG}" \
    "${EXTRA_ARGS[@]}"
