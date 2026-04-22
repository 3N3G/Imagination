#!/usr/bin/env bash
# Eval the 5 AWR checkpoints from the PSF size ablation, 50 episodes each,
# full imagination pipeline (real-time Gemini + embedding). Array 0..4.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_size_ablation_cadence5"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_size_ablation_cadence5"
DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FULL_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb"
HIDDEN_DIM=3072

declare -a TAGS=("full" "top8M" "top4M" "top2M" "top1M")

IDX=${SLURM_ARRAY_TASK_ID:-0}
TAG="${TAGS[$IDX]}"
CKPT_DIR="${CKPT_BASE}/awr_${TAG}"
CKPT="${CKPT_DIR}/final.pth"
STATS="${CKPT_DIR}/hidden_state_stats.npz"
OUT_DIR="${EVAL_BASE}/${TAG}_50ep"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint ${CKPT} not found" >&2
    exit 1
fi

echo "=== PSF size ablation eval — tag=${TAG} ==="

echo "--- validate on held-out train data ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT}" \
    --data-dir "${FULL_DIR}" \
    --file-offset 117 --max-files 8 \
    --hidden-stats "${STATS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0 || true

echo ""
echo "--- live eval 50 episodes ---"
python -m eval.eval_online \
    --checkpoint "${CKPT}" \
    --hidden-stats "${STATS}" \
    --embed-backend gemini_embed \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-episodes 50 \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_psf_size_ablation_${TAG}_50ep"

echo "=== DONE eval ${TAG} ==="
