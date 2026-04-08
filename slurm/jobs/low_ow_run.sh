#!/usr/bin/env bash
# BC+AWR with very low oracle weight — test if near-zero OW recovers baseline return.
# Uses all v3 fixes. Sweeps OW from 0.01 to 0.1.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# OW sweep: 0.01, 0.05, 0.1, 0 (control)
OWS=(    0.01   0.05   0.1    0.0)
TAGS=(   "ow001" "ow005" "ow01" "ow0_control")

OW=${OWS[$ID]}
TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/low_ow_${TAG}"

echo "=== Low OW [${ID}]: ${TAG} (ow=${OW}) ==="

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --entropy-coeff 0.03 \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction 0.05 \
    --max-grad-norm 1.0 \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "low_ow_${TAG}"
