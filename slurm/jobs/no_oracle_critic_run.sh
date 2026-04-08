#!/usr/bin/env bash
# BC+AWR with golden data but critic trained only on PPO data.
# Tests whether critic corruption from golden RTGs (mean 46.8 vs PPO 21.0) causes collapse.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
GOLDEN_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

#  0: OW=0.5, no oracle critic
#  1: OW=1.0, no oracle critic
#  2: OW=0.5, WITH oracle critic (control — should match prior ~2.8 return)
OWS=(   0.5   1.0   0.5)
FLAGS=( "--no-oracle-critic" "--no-oracle-critic" "")
TAGS=(  "nocritic_ow05" "nocritic_ow10" "withcritic_ow05")

OW=${OWS[$ID]}
FLAG=${FLAGS[$ID]}
TAG=${TAGS[$ID]}
SAVE_DIR="${CKPT_BASE}/${TAG}"

echo "=== No-oracle-critic [${ID}]: ${TAG} (ow=${OW}) ==="

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --oracle-data "${GOLDEN_DATA}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout 0.2 \
    --weight-decay 1e-3 \
    --entropy-coeff 0.03 \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction 0.05 \
    --max-grad-norm 1.0 \
    ${FLAG} \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "${TAG}"
