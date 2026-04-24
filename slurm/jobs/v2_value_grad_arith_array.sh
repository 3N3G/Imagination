#!/usr/bin/env bash
# Online eval using value-gradient direction (vs die direction comparison).
#
# Cells:
#   0: A_full   value_grad α=+2
#   1: A_full   value_grad α=-2
#   2: C_grnd   value_grad α=+2
#   3: C_grnd   value_grad α=-2

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0) TAG="predonly";                  TRACK_KEY="predonly_full";       DIR_TRACK="a_full";        ALPHA="2.0"  ;;
    1) TAG="predonly";                  TRACK_KEY="predonly_full";       DIR_TRACK="a_full";        ALPHA="-2.0" ;;
    2) TAG="grounded_predonly_top2M";   TRACK_KEY="grounded_predonly_top2M"; DIR_TRACK="c_grounded_2M"; ALPHA="2.0"  ;;
    3) TAG="grounded_predonly_top2M";   TRACK_KEY="grounded_predonly_top2M"; DIR_TRACK="c_grounded_2M"; ALPHA="-2.0" ;;
    *) echo "ERROR unknown cell ${ID}" >&2; exit 2 ;;
esac

ALPHA_TAG="${ALPHA//./_}"
DIR_PATH="/home/geney/Imagination/probe_results/value_grad_steer/${DIR_TRACK}_value_grad_dir.npy"
CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_value_grad_arith"
RUN_TAG="value_grad_alpha${ALPHA_TAG}"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi
if [ ! -f "${DIR_PATH}" ]; then echo "ERROR: ${DIR_PATH}" >&2; exit 1; fi

echo "=== value_grad arith ID=${ID}  TRACK=${TRACK_KEY}  DIR=${DIR_TRACK}  ALPHA=${ALPHA} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode embed_arith \
    --embed-arith-direction "${DIR_PATH}" \
    --embed-arith-alpha "${ALPHA}" \
    --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${RUN_TAG}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_${RUN_TAG}_30ep"

echo "=== DONE ID=${ID} ==="
