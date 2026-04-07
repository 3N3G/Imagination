#!/usr/bin/env bash
# BC+AWR v7 sweep — ActorCriticAugV2 (deep obs branch + late hidden injection).
# Same top configs as v6, but with the fixed architecture.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# ---- Config grid (6 configs) ----
#  Idx  LR     Beta  Width  OW   Tag
#  0    1e-4   10    512    1.0  lr1e4_ow10
#  1    1e-4   10    512    2.0  lr1e4_ow20
#  2    1e-4   10    512    0.5  lr1e4_ow05
#  3    1e-4   5     512    1.0  lr1e4_beta5
#  4    1e-4   10    1024   1.0  lr1e4_w1024
#  5    3e-4   10    512    1.0  lr3e4_ow10

LRS=(   1e-4  1e-4  1e-4  1e-4  1e-4  3e-4)
BETAS=( 10    10    10    5     10    10)
WIDTHS=(512   512   512   512   1024  512)
OWS=(   1.0   2.0   0.5   1.0   1.0   1.0)
TAGS=(  "lr1e4_ow10"
        "lr1e4_ow20"
        "lr1e4_ow05"
        "lr1e4_beta5"
        "lr1e4_w1024"
        "lr3e4_ow10")

LR=${LRS[$ID]}
BETA=${BETAS[$ID]}
W=${WIDTHS[$ID]}
OW=${OWS[$ID]}
TAG=${TAGS[$ID]}

SAVE_DIR="${CKPT_BASE}/bcawr_v8_${TAG}"

echo "=== Sweep v8 [${ID}]: ${TAG} (arch=V2) ==="
echo "  lr=${LR}, beta=${BETA}, width=${W}, oracle_weight=${OW}"
echo "  save_dir=${SAVE_DIR}"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width "${W}" \
    --lr "${LR}" \
    --awr-beta "${BETA}" \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction 0.10 \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --entropy-coeff 0.01 \
    --weight-decay 1e-4 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "v8_${TAG}"
