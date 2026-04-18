#!/usr/bin/env bash
# v2 50-ep online eval for v2_gemini_emb_freezenone using the live-eval
# pipeline (which loads the same updated predict_state_only_prompt_concise.txt
# at inference time → no train/eval prompt mismatch).
set -euo pipefail

CKPT_DIR="/data/group_data/rl/geney/checkpoints/v2_gemini_emb_freezenone"
EVAL_DIR="/data/group_data/rl/geney/eval_results/v2_gemini_emb_freezenone"

if [ ! -f "${CKPT_DIR}/final.pth" ]; then
    echo "ERROR: ${CKPT_DIR}/final.pth not found." >&2
    exit 1
fi

python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend gemini_embed \
    --hidden-dim 3072 \
    --gemini-model gemini-3-flash-preview \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_v2_gemini_emb_freezenone
