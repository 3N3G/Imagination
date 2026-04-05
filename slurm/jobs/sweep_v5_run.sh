#!/usr/bin/env bash
# BC+AWR v5 sweep — broad hyperparameter search.
# Uses predict-state-only embeddings (gemini-3.1-flash-lite) for train + val.
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

# Training data: predict-state-only, already exists
DATA_DIR="/data/group_data/rl/geney/predict_state_full/final_trajectories"
# Oracle/validation: golden trajectory (also predict-state-only)
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"
VAL_DATA="${ORACLE_DATA}"
CKPT_BASE="/data/group_data/rl/geney/checkpoints"

# ---- Config grid (16 configs) ----
# Axes varied: LR, AWR beta, width, dropout, oracle weight, layernorm
#
#  Idx  LR     Beta  Width Drop  OW   LN?   Tag
#  0    3e-4   10    512   0.0   0.0  yes   baseline_ln
#  1    3e-4   10    512   0.0   0.0  no    baseline_noln
#  2    1e-4   10    512   0.0   1.0  yes   lr1e4_ow10
#  3    3e-4   10    512   0.0   1.0  yes   lr3e4_ow10
#  4    1e-3   10    512   0.0   1.0  yes   lr1e3_ow10
#  5    3e-4   1     512   0.0   1.0  yes   beta1_ow10
#  6    3e-4   5     512   0.0   1.0  yes   beta5_ow10
#  7    3e-4   10    512   0.0   1.0  no    noln_ow10
#  8    3e-4   10    1024  0.0   1.0  yes   w1024_ow10
#  9    3e-4   10    512   0.1   1.0  yes   d01_ow10
# 10    3e-4   10    512   0.0   2.0  yes   ow20
# 11    3e-4   10    512   0.0   0.5  yes   ow05
# 12    1e-4   5     512   0.0   1.0  no    lr1e4_beta5_noln
# 13    1e-4   10    1024  0.0   1.0  yes   lr1e4_w1024
# 14    3e-4   5     1024  0.0   1.0  no    beta5_w1024_noln
# 15    1e-3   5     512   0.1   1.0  yes   lr1e3_beta5_d01

LRS=(   3e-4  3e-4  1e-4  3e-4  1e-3  3e-4  3e-4  3e-4  3e-4  3e-4  3e-4  3e-4  1e-4  1e-4  3e-4  1e-3)
BETAS=( 10    10    10    10    10    1     5     10    10    10    10    10    5     10    5     5)
WIDTHS=(512   512   512   512   512   512   512   512   1024  512   512   512   512   1024  1024  512)
DROPS=( 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.1   0.0   0.0   0.0   0.0   0.0   0.1)
OWS=(   0.0   0.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   1.0   2.0   0.5   1.0   1.0   1.0   1.0)
NOLNS=( 0     1     0     0     0     0     0     1     0     0     0     0     1     0     1     0)
TAGS=(  "baseline_ln"
        "baseline_noln"
        "lr1e4_ow10"
        "lr3e4_ow10"
        "lr1e3_ow10"
        "beta1_ow10"
        "beta5_ow10"
        "noln_ow10"
        "w1024_ow10"
        "d01_ow10"
        "ow20"
        "ow05"
        "lr1e4_beta5_noln"
        "lr1e4_w1024"
        "beta5_w1024_noln"
        "lr1e3_beta5_d01")

LR=${LRS[$ID]}
BETA=${BETAS[$ID]}
W=${WIDTHS[$ID]}
D=${DROPS[$ID]}
OW=${OWS[$ID]}
NOLN=${NOLNS[$ID]}
TAG=${TAGS[$ID]}

SAVE_DIR="${CKPT_BASE}/bcawr_v5_${TAG}"

# Oracle fraction: 0.01 (disabled) when ow=0.0, else 0.10
OF="0.10"
if [ "$OW" = "0.0" ]; then
    OF="0.01"
fi

NOLN_FLAG=""
if [ "$NOLN" = "1" ]; then
    NOLN_FLAG="--no-layernorm"
fi

echo "=== Sweep v5 [${ID}]: ${TAG} ==="
echo "  lr=${LR}, beta=${BETA}, width=${W}, dropout=${D}"
echo "  oracle_weight=${OW}, oracle_frac=${OF}, layernorm=$((1 - NOLN))"
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
    --wandb-name "v5_${TAG}" \
    ${NOLN_FLAG}
