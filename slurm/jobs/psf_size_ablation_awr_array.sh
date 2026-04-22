#!/usr/bin/env bash
# PSF size ablation: augmented AWR pretrain on 5 dataset sizes.
# Array index 0..4 picks: FULL / 8M / 4M / 2M / 1M.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
SUBSET_BASE="${DATA_BASE}/psf_size_ablation_subsets"
FULL_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_cadence5_gemini_emb/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_size_ablation_cadence5"

declare -a TAGS=("full" "top8M" "top4M" "top2M" "top1M")
declare -a DIRS=(
    "${FULL_DIR}"
    "${SUBSET_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb_top8M"
    "${SUBSET_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb_top4M"
    "${SUBSET_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb_top2M"
    "${SUBSET_BASE}/final_trajectories_psf_v2_cadence5_gemini_emb_top1M"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
TAG="${TAGS[$IDX]}"
DATA_DIR="${DIRS[$IDX]}"
SAVE_DIR="${CKPT_BASE}/awr_${TAG}"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} not found" >&2
    exit 1
fi

echo "=== PSF size ablation — tag=${TAG} ==="
echo "  Data:    ${DATA_DIR}"
echo "  Save:    ${SAVE_DIR}"

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 5000 \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim 3072 \
    --lr 3e-4 \
    --awr-beta 10.0 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 100000 \
    --save-freq 25000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.0 \
    --wandb-name "awr_psf_size_ablation_${TAG}" \
    --max-dataset-gb 60

echo "=== DONE awr_psf_size_ablation_${TAG} ==="
