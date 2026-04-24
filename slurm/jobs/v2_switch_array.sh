#!/usr/bin/env bash
# Mid-episode behavioral switch: start episode normally, then flip embedding
# mode to die_v2 / target_descend_v2 / avoid_animals_v2 at step 200.
#
# 4 cells:
#   0: C_grnd   gemini → die_v2  @ step 200    (does behavior collapse fast?)
#   1: C_grnd   gemini → target_descend_v2 @ 200  (does descend rate spike post-switch?)
#   2: A_full   gemini → die_v2  @ step 200
#   3: C_grnd   gemini → avoid_animals_v2 @ 200

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0) TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"; SWITCH_MODE="die_v2"; SWITCH_STEP=200 ;;
    1) TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"; SWITCH_MODE="target_descend_v2"; SWITCH_STEP=200 ;;
    2) TAG="predonly";                TRACK_KEY="predonly_full";          SWITCH_MODE="die_v2"; SWITCH_STEP=200 ;;
    3) TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"; SWITCH_MODE="avoid_animals_v2"; SWITCH_STEP=200 ;;
    *) echo "ERROR unknown cell ${ID}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_switch"
RUN_TAG="switch_to_${SWITCH_MODE}_at${SWITCH_STEP}"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== switch ID=${ID}  TRACK=${TRACK_KEY}  SWITCH=${SWITCH_MODE} @ step ${SWITCH_STEP} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode gemini \
    --switch-mode "${SWITCH_MODE}" --switch-step "${SWITCH_STEP}" \
    --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${RUN_TAG}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_${RUN_TAG}_30ep"

echo "=== DONE ID=${ID} ==="
