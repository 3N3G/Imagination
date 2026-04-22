#!/usr/bin/env bash
# Track A: 50-ep live eval of freezenone predonly checkpoint.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly"
DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"

CKPT="${CKPT_BASE}/freezenone/final.pth"
STATS="${CKPT_BASE}/freezenone/hidden_state_stats.npz"
FULL_DIR="${DATA_BASE}/final_trajectories_psf_v2_cadence5_predonly_gemini_emb"
OUT_DIR="${EVAL_BASE}/freezenone_50ep"
HIDDEN_DIM=3072

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint ${CKPT} not found" >&2
    exit 1
fi

echo "=== Track A: predonly freezenone 50-ep eval ==="

echo "--- validate on held-out train data ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT}" \
    --data-dir "${FULL_DIR}" \
    --file-offset 117 --max-files 8 \
    --hidden-stats "${STATS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0 || true

echo ""
echo "--- live eval 50 episodes (predonly inference) ---"
python -m eval.eval_online \
    --checkpoint "${CKPT}" \
    --hidden-stats "${STATS}" \
    --embed-backend gemini_embed \
    --hidden-dim "${HIDDEN_DIM}" \
    --extract-prediction-only \
    --num-episodes 50 \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_track_a_predonly_freezenone_50ep"

echo "=== DONE Track A eval ==="
