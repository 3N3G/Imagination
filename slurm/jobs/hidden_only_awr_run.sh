#!/usr/bin/env bash
# Hidden-only baseline: policy takes ONLY the imagination embedding, no obs.
# Pure AWR (oracle loss weight 0). Tests whether Gemini narratives alone
# carry enough signal to drive a competent policy.
#
# Compare to:
#   unaug ActorCritic (obs only):              18.38 (strong baseline)
#   awr_aug_debug (obs + hidden, pure AWR):    16.30 ± 3.61
set -euo pipefail

DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/hidden_only_awr"
EVAL_DIR="/data/group_data/rl/geney/eval_results/hidden_only_awr"

echo "======================================================================"
echo "PHASE 1: Train hidden_only pure AWR (100K from-scratch)"
echo "  Model: ActorCriticHiddenOnly (no obs branch)"
echo "  Loss:  AWR only (oracle-loss-weight 0)"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${CKPT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --hidden-mode real \
    --layer-width 512 \
    --lr 3e-4 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --arch-hidden-only \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.0 \
    --total-steps 100000 \
    --save-freq 25000 \
    --wandb-name hidden_only_awr \
    --max-dataset-gb 30 \
    --no-wandb

echo ""
echo "======================================================================"
echo "PHASE 2: Held-out validation (last 8 files)"
echo "======================================================================"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${DATA_DIR}" \
    --file-offset 126 --max-files 8 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --arch-hidden-only \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 3: Golden validation"
echo "======================================================================"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir /data/group_data/rl/geney/oracle_pipeline/final_trajectories \
    --file-offset 0 --max-files 1 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --arch-hidden-only \
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 4: 50-ep online eval"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --arch-hidden-only \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_hidden_only_awr \
    --no-wandb

echo ""
echo "======================================================================"
echo "ALL DONE — hidden_only_awr"
echo "======================================================================"
