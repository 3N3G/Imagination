#!/usr/bin/env bash
# Phases 4-5 only: Validate + Eval for already-trained gemini_emb checkpoint.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_gemini_emb"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/awr_psf_gemini_emb"
EVAL_DIR="/data/group_data/rl/geney/eval_results/awr_psf_gemini_emb_50ep"
HIDDEN_DIM=3072

echo "======================================================================"
echo "PHASE 4: Validate (held-out training)"
echo "======================================================================"
echo "--- Held-out training data (last 8 files) ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${FINAL_DIR}" \
    --file-offset 117 --max-files 8 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout 0.0

echo ""
echo "--- Oracle (golden) data: SKIPPED (oracle=4096-dim Qwen3-8B; gemini_emb=3072-dim) ---"

echo ""
echo "======================================================================"
echo "PHASE 5: Live eval (50 episodes)"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --embed-backend gemini_embed \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_awr_psf_gemini_emb_50ep

echo ""
echo "======================================================================"
echo "ALL DONE — gemini_emb eval"
echo "======================================================================"
