#!/usr/bin/env bash
# BC+AWR v6 sweep — PSF-consistent golden data, top configs from v5.
# Key change: oracle data now labelled with predict-only (matching training data).
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# Training data: predict-state-only
DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
# Oracle: golden trajectory re-labelled with predict-only
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final/trajectories_000000.npz"
# Validation: held-out test trajectory (predict-only, never seen during training)
VAL_DATA="/data/group_data/rl/geney/oracle_pipeline/test_final/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# ---- Config grid (6 configs) ----
# Based on v5 results: LR=1e-4 dominated, LayerNorm essential, moderate oracle weight best.
#
#  Idx  LR     Beta  Width Drop  OW   Tag                  Rationale
#  0    1e-4   10    512   0.0   1.0  lr1e4_ow10           v5 best (Δ=+0.47)
#  1    1e-4   10    512   0.0   2.0  lr1e4_ow20           best LR + high OW
#  2    1e-4   10    512   0.0   0.5  lr1e4_ow05           best LR + low OW
#  3    1e-4   5     512   0.0   1.0  lr1e4_beta5          best LR + medium beta
#  4    1e-4   10    1024  0.0   1.0  lr1e4_w1024          best LR + wide network
#  5    3e-4   10    512   0.0   1.0  lr3e4_ow10           default LR reference

LRS=(   1e-4  1e-4  1e-4  1e-4  1e-4  3e-4)
BETAS=( 10    10    10    5     10    10)
WIDTHS=(512   512   512   512   1024  512)
DROPS=( 0.0   0.0   0.0   0.0   0.0   0.0)
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
D=${DROPS[$ID]}
OW=${OWS[$ID]}
TAG=${TAGS[$ID]}

SAVE_DIR="${CKPT_BASE}/bcawr_v6_${TAG}"

# Oracle fraction: 0.10 for all (all have oracle weight > 0)
OF="0.10"

echo "=== Sweep v6 [${ID}]: ${TAG} ==="
echo "  lr=${LR}, beta=${BETA}, width=${W}, dropout=${D}"
echo "  oracle_weight=${OW}, oracle_frac=${OF}"
echo "  oracle_data=${ORACLE_DATA}"
echo "  val_data=${VAL_DATA}"
echo "  save_dir=${SAVE_DIR}"
echo ""

python -m offline_rl.train_awr_weighted_v2 \
    --save-dir "${SAVE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --layer-width "${W}" \
    --lr "${LR}" \
    --awr-beta "${BETA}" \
    --dropout "${D}" \
    --oracle-loss-weight "${OW}" \
    --oracle-fraction "${OF}" \
    --oracle-data "${ORACLE_DATA}" \
    --val-data "${VAL_DATA}" \
    --val-freq 5000 \
    --entropy-coeff 0.01 \
    --weight-decay 1e-4 \
    --total-steps 100000 \
    --wandb-name "v6_${TAG}"
