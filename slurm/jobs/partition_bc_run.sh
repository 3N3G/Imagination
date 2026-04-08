#!/usr/bin/env bash
# Partition test: AWR on files 0-119, BC on file 120 (same PPO distribution).
# Tests whether BC+AWR objective itself hurts, independent of data distribution.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
BC_DATA="${DATA_DIR}/trajectories_000120.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# Sweep OW with everything else fixed:
#  0: OW=0.0   (pure AWR control, no BC)
#  1: OW=0.1   (light BC)
#  2: OW=0.5   (moderate BC)
#  3: OW=1.0   (heavy BC)
OWS=(   0.0   0.1   0.5   1.0)
TAGS=(  "part_ow0" "part_ow01" "part_ow05" "part_ow10")

OW=${OWS[$ID]}
TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "=== Partition BC [${ID}]: ${TAG} (ow=${OW}) ==="

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --max-files 120 \
    --oracle-data "${BC_DATA}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --entropy-coeff 0.03 \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction 0.05 \
    --max-grad-norm 1.0 \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "${TAG}"
