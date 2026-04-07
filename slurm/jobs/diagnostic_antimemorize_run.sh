#!/usr/bin/env bash
# Anti-memorization BC+AWR diagnostic.
# Aggressive regularization to prevent oracle dataset memorization.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# 4 configs testing different anti-memorization strategies:
#
#  0: high_dropout     — dropout=0.3, wd=1e-3, ent=0.05, ow=1.0, of=0.05
#  1: low_oracle       — dropout=0.1, wd=1e-4, ent=0.01, ow=0.2, of=0.03
#  2: aggressive_reg   — dropout=0.5, wd=1e-2, ent=0.1,  ow=1.0, of=0.10
#  3: moderate_combo   — dropout=0.2, wd=1e-3, ent=0.03, ow=0.5, of=0.05

DROPS=(  0.3   0.1   0.5   0.2)
WDS=(    1e-3  1e-4  1e-2  1e-3)
ENTS=(   0.05  0.01  0.1   0.03)
OWS=(    1.0   0.2   1.0   0.5)
OFS=(    0.05  0.03  0.10  0.05)
TAGS=(   "high_dropout" "low_oracle" "aggressive_reg" "moderate_combo")

D=${DROPS[$ID]}
WD=${WDS[$ID]}
ENT=${ENTS[$ID]}
OW=${OWS[$ID]}
OF=${OFS[$ID]}
TAG=${TAGS[$ID]}

SAVE_DIR="${CKPT_BASE}/antimemorize_${TAG}"

echo "=== Anti-memorize [${ID}]: ${TAG} ==="
echo "  dropout=${D}, wd=${WD}, ent=${ENT}, ow=${OW}, of=${OF}"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width 1024 \
    --lr 1e-4 \
    --awr-beta 10 \
    --dropout "${D}" \
    --weight-decay "${WD}" \
    --entropy-coeff "${ENT}" \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction "${OF}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --total-steps 100000 \
    --arch-v2 \
    --wandb-name "antimemorize_${TAG}"
