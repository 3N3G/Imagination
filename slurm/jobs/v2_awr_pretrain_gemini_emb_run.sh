#!/usr/bin/env bash
# v2 Step 1: pretrain AWR-only on the v2 PSF gemini_emb shards.
# Output: checkpoints/awr_psf_v2_gemini_emb/  (used as init for freezenone fine-tune)
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
DATA_DIR="${DATA_BASE}/final_trajectories_psf_v2_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_gemini_emb/trajectories_000000.npz"
SAVE_DIR="/data/group_data/rl/geney/checkpoints/awr_psf_v2_gemini_emb"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: ${DATA_DIR} not found — run v2_embed_merge_psf_gemini_emb_run.sh first." >&2
    exit 1
fi

echo "=== v2 AWR pretrain on gemini_emb shards ==="
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
    --wandb-name awr_psf_v2_gemini_emb \
    --max-dataset-gb 60

echo ""
echo "=== DONE awr_psf_v2_gemini_emb ==="
