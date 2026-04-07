#!/usr/bin/env bash
# Gated imagination BC+AWR run.
# The model learns a 0/1 gate deciding whether to use imagination per-sample.
set -euo pipefail

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
SAVE_DIR="/data/group_data/rl/geney/checkpoints/gated_v1"

echo "=== Gated Imagination BC+AWR ==="
echo "  Config: lr1e-4, w1024, ow=1.0, dropout=0.2, wd=1e-3, ent=0.03"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --entropy-coeff 0.03 \
    --oracle-loss-weight 1.0 \
    --oracle-fraction 0.10 \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 2500 \
    --total-steps 100000 \
    --arch-gated \
    --wandb-name "gated_v1"
