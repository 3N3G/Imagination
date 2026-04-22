#!/usr/bin/env bash
# In-distribution semantic probe on Track A predonly freezenone.
# No Gemini calls; just forward passes on training-distribution (obs, hidden) pairs.
set -euo pipefail

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_predonly/freezenone"
DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories_psf_v2_cadence5_predonly_gemini_emb"
OUT_DIR="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_predonly_semantic/in_distribution_probe"

python -m tools.in_distribution_semantic_probe \
    --checkpoint "${CKPT_BASE}/final.pth" \
    --hidden-stats "${CKPT_BASE}/hidden_state_stats.npz" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}" \
    --num-samples 500 \
    --num-cf-pairs 3 \
    --hidden-dim 3072
