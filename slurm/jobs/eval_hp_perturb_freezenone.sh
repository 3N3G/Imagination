#!/usr/bin/env bash
# HP/Food perturbation eval for the phase1 freezenone winner (gemini_emb).
set -euo pipefail

CKPT_DIR="/data/group_data/rl/geney/checkpoints/phase1_gemini_emb_freezenone"
OUT_DIR="/data/group_data/rl/geney/eval_results/phase1_gemini_emb_freezenone_hp_perturb"

echo "===================================================================="
echo "hp_perturb phase1_gemini_emb_freezenone  backend=gemini_embed  dim=3072"
echo "===================================================================="

python -m eval.eval_hp_perturbation \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --embed-backend "gemini_embed" \
    --hidden-dim 3072 \
    --num-episodes 10 \
    --probe-every 1 \
    --output-dir "${OUT_DIR}"
