#!/usr/bin/env bash
# Re-run validate_awr for the 5 PSF size-ablation checkpoints with enough
# memory to avoid OOM. Array 0..4 mirrors psf_size_ablation_eval_array.sh.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_size_ablation_cadence5"
DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FULL_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb"
HIDDEN_DIM=3072

declare -a TAGS=("full" "top8M" "top4M" "top2M" "top1M")

IDX=${SLURM_ARRAY_TASK_ID:-0}
TAG="${TAGS[$IDX]}"
CKPT_DIR="${CKPT_BASE}/awr_${TAG}"
CKPT="${CKPT_DIR}/final.pth"
STATS="${CKPT_DIR}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint ${CKPT} not found" >&2
    exit 1
fi

echo "=== PSF size ablation validate — tag=${TAG} ==="

python -m eval.validate_awr \
    --checkpoint "${CKPT}" \
    --data-dir "${FULL_DIR}" \
    --file-offset 117 --max-files 8 \
    --hidden-stats "${STATS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0

echo "=== DONE validate ${TAG} ==="
