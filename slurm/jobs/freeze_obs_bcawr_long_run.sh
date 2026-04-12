#!/usr/bin/env bash
# Longer freeze_obs_bcawr run (200K steps vs original 50K) to test whether
# extended training can push past unaug baseline (18.38).
# Config matches freeze_bc_run.sh ID=1 (freeze obs_branch, BC+AWR).
set -euo pipefail

DATA_DIR="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
PRETRAINED="/data/group_data/rl/geney/checkpoints/awr_aug_debug/final.pth"
CKPT_DIR="/data/group_data/rl/geney/checkpoints/freeze_obs_bcawr_long"
EVAL_DIR="/data/group_data/rl/geney/eval_results/freeze_obs_bcawr_long"

echo "======================================================================"
echo "PHASE 1: Train freeze_obs_bcawr_long (200K steps)"
echo "======================================================================"
python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${CKPT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --pretrained-checkpoint "${PRETRAINED}" \
    --freeze-mode obs_branch \
    --hidden-mode real \
    --layer-width 512 \
    --lr 1e-4 \
    --entropy-coeff 0.01 \
    --max-grad-norm 1.0 \
    --oracle-fraction 0.05 \
    --oracle-loss-weight 0.5 \
    --total-steps 200000 \
    --save-freq 25000 \
    --wandb-name freeze_obs_bcawr_long \
    --max-dataset-gb 30

echo ""
echo "======================================================================"
echo "PHASE 2: Held-out validation (last 8 files)"
echo "======================================================================"
python -m eval.validate_awr \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --data-dir "${DATA_DIR}" \
    --file-offset 126 --max-files 8 \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
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
    --dropout 0.0

echo ""
echo "======================================================================"
echo "PHASE 4: 50-ep online eval"
echo "======================================================================"
python -m eval.eval_online \
    --checkpoint "${CKPT_DIR}/final.pth" \
    --hidden-stats "${CKPT_DIR}/hidden_state_stats.npz" \
    --layer-width 512 \
    --num-episodes 50 \
    --output-dir "${EVAL_DIR}" \
    --wandb-name eval_freeze_obs_bcawr_long

echo ""
echo "======================================================================"
echo "ALL DONE — freeze_obs_bcawr_long"
echo "======================================================================"
