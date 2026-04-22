#!/usr/bin/env bash
# Rerun Track A predonly multistep direction CF after 503 crash of array task 4.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_semantic"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

python -m eval.eval_direction_counterfactual_multistep \
    --checkpoint "${CKPT}" \
    --hidden-stats "${STATS}" \
    --embed-backend gemini_embed \
    --hidden-dim 3072 \
    --extract-prediction-only \
    --intervention-steps 0,75,150,300 \
    --num-episodes 30 \
    --output-dir "${EVAL_BASE}/direction_cf_multistep"
