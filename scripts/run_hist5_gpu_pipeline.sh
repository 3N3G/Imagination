#!/usr/bin/env bash
# GPU pipeline for history-5: embed → merge → train → validate → eval
set -euo pipefail

FILTERED_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/filtered_trajectories"
HIST5_BASE="/data/group_data/rl/geney/predict_history_k5"
GEMINI_DIR="$HIST5_BASE/gemini_labels"
EMBED_DIR="$HIST5_BASE/embeddings"
FINAL_DIR="$HIST5_BASE/final_trajectories"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/awr_hist5"
EVAL_DIR="/data/group_data/rl/geney/eval_results/awr_hist5_50ep"

echo "======================================================================"
echo "PHASE 2: Qwen embedding"
echo "======================================================================"
python -m pipeline.embed \
    --gemini-dir "$GEMINI_DIR" \
    --output-dir "$EMBED_DIR"

echo ""
echo "======================================================================"
echo "PHASE 3: Merge"
echo "======================================================================"
python -m pipeline.merge \
    --filtered-dir "$FILTERED_DIR" \
    --gemini-dir "$GEMINI_DIR" \
    --embed-dir "$EMBED_DIR" \
    --output-dir "$FINAL_DIR"

echo ""
echo "======================================================================"
echo "PHASE 4: Train pure AWR (no BC) on history-5 data"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --data-dir "$FINAL_DIR" \
    --save-dir "$CKPT_DIR" \
    --wandb-name awr_hist5 \
    --oracle-fraction 0.0 \
    --oracle-loss-weight 0.0 \
    --max-grad-norm 1.0 \
    --total-steps 100000 \
    --max-dataset-gb 60

echo ""
echo "======================================================================"
echo "PHASE 5: Action prediction validation"
echo "======================================================================"
# Held-out training data (files 126+)
echo "--- Held-out training data ---"
python -m eval.validate_awr \
    --checkpoint "$CKPT_DIR/final.pth" \
    --data-dir "$FINAL_DIR" \
    --file-offset 126 --max-files 32 \
    --hidden-stats "$CKPT_DIR/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "--- Oracle (golden) data ---"
python -m eval.validate_awr \
    --checkpoint "$CKPT_DIR/final.pth" \
    --data-dir /data/group_data/rl/geney/oracle_pipeline/final_trajectories \
    --file-offset 0 --max-files 1 \
    --hidden-stats "$CKPT_DIR/hidden_state_stats.npz" \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 6: Live eval (50 episodes)"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "$CKPT_DIR/final.pth" \
    --hidden-stats "$CKPT_DIR/hidden_state_stats.npz" \
    --num-episodes 50 \
    --output-dir "$EVAL_DIR" \
    --wandb-name eval_awr_hist5_50ep

echo ""
echo "======================================================================"
echo "ALL DONE"
echo "======================================================================"
