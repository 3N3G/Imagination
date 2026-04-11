#!/usr/bin/env bash
# Phases 3-5 only for qwen3gen (embed+merge already done).
# Restart after OOM: using --max-dataset-gb 30 for smaller shards.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_qwen3gen"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/awr_psf_qwen3gen"
EVAL_DIR="/data/group_data/rl/geney/eval_results/awr_psf_qwen3gen_50ep"

echo "======================================================================"
echo "PHASE 3: Train pure AWR (no BC) — qwen3gen embeddings"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --data-dir "${FINAL_DIR}" \
    --save-dir "${CKPT_DIR}" \
    --wandb-name awr_psf_qwen3gen \
    --oracle-fraction 0.0 \
    --oracle-loss-weight 0.0 \
    --max-grad-norm 1.0 \
    --total-steps 100000 \
    --max-dataset-gb 30

echo ""
echo "======================================================================"
echo "PHASE 4: Validate (held-out training + oracle data)"
echo "======================================================================"
echo "--- Held-out training data (files 126+) ---"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${FINAL_DIR}" \
    --file-offset 126 --max-files 8 \
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
    --embed-backend qwen3_gen \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_awr_psf_qwen3gen_50ep

echo ""
echo "======================================================================"
echo "ALL DONE — qwen3gen (phases 3-5)"
echo "======================================================================"
