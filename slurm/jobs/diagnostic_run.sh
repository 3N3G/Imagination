#!/usr/bin/env bash
# Heavily-instrumented diagnostic BC+AWR run.
# Purpose: determine if the model uses imagination semantically or as a source tag.
set -euo pipefail

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
SAVE_DIR="/data/group_data/rl/geney/checkpoints/diagnostic_v9"

echo "=== Diagnostic BC+AWR Run ==="
echo "  Config: lr1e4, w1024, ow1.0, beta10, arch-v2"
echo "  Full instrumentation: counterfactual val, gradient conflict, separability"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --oracle-loss-weight 1.0 \
    --oracle-fraction 0.10 \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 2500 \
    --entropy-coeff 0.01 \
    --weight-decay 1e-4 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "diagnostic_v9"
