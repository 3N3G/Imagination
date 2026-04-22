#!/usr/bin/env bash
# Track A: freezenone BC+AWR fine-tune on predonly-embedded PSF data.
# Same Apr-17 winning recipe (β=30, ofrac=0.05, ow=0.5, freeze=none).
# Init from Track A AWR pretrain (checkpoints/psf_v2_cadence5_predonly/awr/final.pth).
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly"

DATA_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_predonly_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_cadence5_gemini_emb/trajectories_000000.npz"
PRETRAINED="${CKPT_BASE}/awr/final.pth"
SAVE_DIR="${CKPT_BASE}/freezenone"

if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: ${PRETRAINED} not found — run track_a_predonly_awr.sh first." >&2
    exit 1
fi

echo "=== Track A: freezenone BC+AWR fine-tune on predonly data ==="
echo "  Data:        ${DATA_DIR}"
echo "  Pretrained:  ${PRETRAINED}"
echo "  Save:        ${SAVE_DIR}"

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${ORACLE_DATA}" \
    --val-freq 2500 \
    --pretrained-checkpoint "${PRETRAINED}" \
    --freeze-mode none \
    --hidden-mode real \
    --layer-width 512 \
    --hidden-dim 3072 \
    --lr 1e-4 \
    --awr-beta 30.0 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --total-steps 50000 \
    --save-freq 10000 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --wandb-name track_a_predonly_freezenone \
    --max-dataset-gb 60

echo ""
echo "=== DONE Track A freezenone ==="
