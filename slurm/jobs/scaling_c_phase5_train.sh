#!/usr/bin/env bash
# SCALING_C Phase 5 — train AWR + BC+AWR on the new top-4M data.
# Same recipe as C_grounded_2M (Track C / Apr-22 winner):
#   Stage 0: AWR pretrain  (train_awr_weighted_v2, β=10, ow=0.0, lr=3e-4, 100k steps)
#   Stage 1: BC+AWR freezenone (β=30, ow=0.5, lr=1e-4, 50k steps, init from Stage 0)
# Stage 2 is the eval (Phase 6, separate script).
#
# Stage selected by SLURM array index 0|1.

set -euo pipefail

ID="${SLURM_ARRAY_TASK_ID:-${1:-}}"
if [ -z "${ID}" ]; then
    echo "Usage: pass stage as first arg (0=AWR, 1=BC+AWR) or via SLURM array" >&2
    exit 2
fi

# Phase 4 wrote final data to user_data; oracle data is on group_data.
# Checkpoints land back on group_data (now 68G free again as of 02:55 EDT)
# to keep user_data from filling — checkpoints can be a few GB each.
DATA_DIR="/data/user_data/geney/scaling_c_data/final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M"
ORACLE_DATA="/data/group_data/rl/geney/oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb/trajectories_000000.npz"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_top4M"

case "${ID}" in
    0)
        SAVE_DIR="${CKPT_BASE}/awr"
        mkdir -p "${SAVE_DIR}"
        echo "=== SCALING_C Phase 5 stage 0: AWR pretrain ==="
        python -m offline_rl.train_awr_weighted_v2 \
            --save-dir "${SAVE_DIR}" --data-dir "${DATA_DIR}" \
            --oracle-data "${ORACLE_DATA}" --val-data "${ORACLE_DATA}" \
            --val-freq 5000 --hidden-mode real --layer-width 512 --hidden-dim 3072 \
            --lr 3e-4 --awr-beta 10.0 --entropy-coeff 0.01 --max-grad-norm 1.0 \
            --total-steps 100000 --save-freq 25000 \
            --oracle-fraction 0.05 --oracle-loss-weight 0.0 \
            --wandb-name "awr_psf_v3_pporn_1e8_grounded_top4M" --max-dataset-gb 120
        ;;
    1)
        SAVE_DIR="${CKPT_BASE}/freezenone"
        PRETRAINED="${CKPT_BASE}/awr/final.pth"
        if [ ! -f "${PRETRAINED}" ]; then
            echo "ERROR: ${PRETRAINED} not found — Stage 0 not complete" >&2
            exit 1
        fi
        mkdir -p "${SAVE_DIR}"
        echo "=== SCALING_C Phase 5 stage 1: BC+AWR freezenone ==="
        python -m offline_rl.train_awr_weighted_v2 \
            --save-dir "${SAVE_DIR}" --data-dir "${DATA_DIR}" \
            --oracle-data "${ORACLE_DATA}" --val-data "${ORACLE_DATA}" \
            --val-freq 2500 --pretrained-checkpoint "${PRETRAINED}" \
            --freeze-mode none --hidden-mode real --layer-width 512 --hidden-dim 3072 \
            --lr 1e-4 --awr-beta 30.0 --entropy-coeff 0.01 --max-grad-norm 1.0 \
            --total-steps 50000 --save-freq 10000 \
            --oracle-fraction 0.05 --oracle-loss-weight 0.5 \
            --wandb-name "freezenone_psf_v3_pporn_1e8_grounded_top4M" --max-dataset-gb 120
        ;;
    *)
        echo "unknown stage: ${ID}" >&2
        exit 2
        ;;
esac
