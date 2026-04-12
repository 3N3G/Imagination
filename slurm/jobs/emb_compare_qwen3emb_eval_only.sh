#!/usr/bin/env bash
# Phases 4-5 only for qwen3emb (embed+merge+train already done).
# Restart after validate crash (file-offset 126 out of range for 125-file dataset).
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_qwen3emb"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/awr_psf_qwen3emb"
EVAL_DIR="/data/group_data/rl/geney/eval_results/awr_psf_qwen3emb_50ep"

echo "======================================================================"
echo "PHASE 4: Validate (held-out training + oracle data)"
echo "======================================================================"
echo "--- Held-out training data (last 8 files) ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${FINAL_DIR}" \
    --file-offset 117 --max-files 8 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "--- Oracle (golden) data ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir /data/group_data/rl/geney/oracle_pipeline/final_trajectories \
    --file-offset 0 --max-files 1 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 5: Live eval (50 episodes)"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --embed-backend qwen3_embed \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_awr_psf_qwen3emb_50ep

echo ""
echo "======================================================================"
echo "ALL DONE — qwen3emb validate + eval"
echo "======================================================================"
