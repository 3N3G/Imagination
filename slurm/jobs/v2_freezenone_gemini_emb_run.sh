#!/usr/bin/env bash
# v2 Step 2: freezenone BC+AWR fine-tune on the v2 gemini_emb data.
# This is the Phase-1 winning recipe (β=30, ofrac=0.05, ow=0.5, freeze=none)
# from Apr-17 — only config that beat the gemini_emb β=30 baseline.
# Init from awr_psf_v2_gemini_emb (Step 1).
# Output: checkpoints/v2_gemini_emb_freezenone/
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

DATA_DIR="${DATA_BASE}/final_trajectories_psf_v2_gemini_emb"
ORACLE_DATA="${ORACLE_BASE}/predict_only_final_v2_gemini_emb/trajectories_000000.npz"
PRETRAINED="${CKPT_BASE}/awr_psf_v2_gemini_emb/final.pth"
SAVE_DIR="${CKPT_BASE}/v2_gemini_emb_freezenone"

if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: ${PRETRAINED} not found — run v2_awr_pretrain_gemini_emb_run.sh first." >&2
    exit 1
fi

echo "=== v2 freezenone BC+AWR fine-tune on gemini_emb ==="
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
    --wandb-name v2_gemini_emb_freezenone \
    --max-dataset-gb 60

echo ""
echo "=== DONE v2_gemini_emb_freezenone ==="
