#!/usr/bin/env bash
# Embedding comparison: Qwen3-Embedding (dedicated embedding model, last-token pool) on PSF data.
# Embed → Merge → Train pure AWR → Validate → Eval
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf"
EMBED_DIR="${DATA_BASE}/embeddings_psf_qwen3emb"
FINAL_DIR="${DATA_BASE}/final_trajectories_psf_qwen3emb"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/awr_psf_qwen3emb"
EVAL_DIR="/data/group_data/rl/geney/eval_results/awr_psf_qwen3emb_50ep"

echo "======================================================================"
echo "PHASE 1: Qwen3-Embedding embed (PSF labels)"
echo "======================================================================"
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend qwen3_embed \
    --batch-size 32

echo ""
echo "======================================================================"
echo "PHASE 2: Merge"
echo "======================================================================"
python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}"

echo ""
echo "======================================================================"
echo "PHASE 3: Train pure AWR (no BC)"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --data-dir "${FINAL_DIR}" \
    --save-dir "${CKPT_DIR}" \
    --wandb-name awr_psf_qwen3emb \
    --oracle-fraction 0.0 \
    --oracle-loss-weight 0.0 \
    --max-grad-norm 1.0 \
    --total-steps 100000 \
    --max-dataset-gb 60

echo ""
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
echo "ALL DONE — qwen3emb"
echo "======================================================================"
