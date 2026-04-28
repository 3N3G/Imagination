#!/usr/bin/env bash
# Parameterized SCALING_C variant trainer. Reads env vars set by submitter:
#   VARIANT_TAG     (string used in CKPT_BASE + wandb names)
#   STAGE           (0 = AWR pretrain, 1 = BC+AWR freezenone)
#   DATA_DIR        (override data dir; default = the new top-4M PPO-RNN data)
#   PRETRAINED      (stage 1: pretrained checkpoint path; defaults to ${CKPT_BASE}/awr/final.pth)
#   AWR_BETA_S0     (default 10.0 — stage 0)
#   AWR_BETA_S1     (default 30.0 — stage 1)
#   ORACLE_LOSS_WEIGHT_S1 (default 0.5 — stage 1)
# Other args match C_grounded_2M recipe exactly: lr 3e-4/1e-4, hidden_dim 3072,
# layer_width 512, oracle_fraction 0.05, entropy_coeff 0.01, max_grad_norm 1.0,
# total_steps 100000/50000, save_freq 25000/10000.
#
# Auto-picks COMBINED oracle data file if it exists.

set -euo pipefail

if [ -z "${VARIANT_TAG:-}" ]; then
    echo "ERROR: set VARIANT_TAG env var" >&2; exit 2
fi
if [ -z "${STAGE:-}" ]; then
    echo "ERROR: set STAGE env var (0|1)" >&2; exit 2
fi

DATA_DIR="${DATA_DIR:-/data/user_data/geney/scaling_c_data/final_trajectories_psf_v3_cadence5_grounded_predonly_gemini_emb_top4M}"

ORACLE_COMBINED="/data/group_data/rl/geney/oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb_combined/trajectories_000000.npz"
ORACLE_LEGACY="/data/group_data/rl/geney/oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb/trajectories_000000.npz"
if [ -f "${ORACLE_COMBINED}" ]; then
    ORACLE_DATA="${ORACLE_COMBINED}"
    echo "Using COMBINED oracle data (37 trajs): ${ORACLE_DATA}"
else
    ORACLE_DATA="${ORACLE_LEGACY}"
    echo "WARNING: combined oracle file not found, falling back to LEGACY (25 trajs): ${ORACLE_DATA}"
fi

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}"

AWR_BETA_S0="${AWR_BETA_S0:-10.0}"
AWR_BETA_S1="${AWR_BETA_S1:-30.0}"
ORACLE_LOSS_WEIGHT_S1="${ORACLE_LOSS_WEIGHT_S1:-0.5}"

case "${STAGE}" in
    0)
        SAVE_DIR="${CKPT_BASE}/awr"
        mkdir -p "${SAVE_DIR}"
        echo "=== Variant ${VARIANT_TAG} stage 0: AWR pretrain (β=${AWR_BETA_S0}) ==="
        echo "  DATA_DIR: ${DATA_DIR}"
        echo "  ORACLE:   ${ORACLE_DATA}"
        python -m offline_rl.train_awr_weighted_v2 \
            --save-dir "${SAVE_DIR}" --data-dir "${DATA_DIR}" \
            --oracle-data "${ORACLE_DATA}" --val-data "${ORACLE_DATA}" \
            --val-freq 5000 --hidden-mode real --layer-width 512 --hidden-dim 3072 \
            --lr 3e-4 --awr-beta "${AWR_BETA_S0}" --entropy-coeff 0.01 --max-grad-norm 1.0 \
            --total-steps 100000 --save-freq 25000 \
            --oracle-fraction 0.05 --oracle-loss-weight 0.0 \
            --wandb-name "awr_psf_v3_pporn_1e8_grounded_${VARIANT_TAG}" --max-dataset-gb 200
        ;;
    1)
        SAVE_DIR="${CKPT_BASE}/freezenone"
        PRETRAINED="${PRETRAINED:-${CKPT_BASE}/awr/final.pth}"
        if [ ! -f "${PRETRAINED}" ]; then
            echo "ERROR: ${PRETRAINED} not found — set PRETRAINED env var or run stage 0 first" >&2
            exit 1
        fi
        mkdir -p "${SAVE_DIR}"
        echo "=== Variant ${VARIANT_TAG} stage 1: BC+AWR freezenone (β=${AWR_BETA_S1}, ow=${ORACLE_LOSS_WEIGHT_S1}) ==="
        echo "  DATA_DIR:   ${DATA_DIR}"
        echo "  ORACLE:     ${ORACLE_DATA}"
        echo "  PRETRAINED: ${PRETRAINED}"
        python -m offline_rl.train_awr_weighted_v2 \
            --save-dir "${SAVE_DIR}" --data-dir "${DATA_DIR}" \
            --oracle-data "${ORACLE_DATA}" --val-data "${ORACLE_DATA}" \
            --val-freq 2500 --pretrained-checkpoint "${PRETRAINED}" \
            --freeze-mode none --hidden-mode real --layer-width 512 --hidden-dim 3072 \
            --lr 1e-4 --awr-beta "${AWR_BETA_S1}" --entropy-coeff 0.01 --max-grad-norm 1.0 \
            --total-steps 50000 --save-freq 10000 \
            --oracle-fraction 0.05 --oracle-loss-weight "${ORACLE_LOSS_WEIGHT_S1}" \
            --wandb-name "freezenone_psf_v3_pporn_1e8_grounded_${VARIANT_TAG}" --max-dataset-gb 200
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2; exit 2
        ;;
esac
