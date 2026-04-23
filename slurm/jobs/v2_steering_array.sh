#!/usr/bin/env bash
# v2 steering probe — 50-ep eval across 4 tracks × 2 steering variants.
# Variants reframe the "good algorithm" so water / cows are navigationally
# opaque; Gemini emits positive direction statements with no "avoid" /
# "instead of" / "away from" phrasing (probe confirmed 0% negation).
#
# Array index → (track, variant):
#   0: A_full  / avoid_water_v2   (psf_v2_cadence5_predonly)
#   1: A_full  / avoid_animals_v2
#   2: A_2M    / avoid_water_v2   (psf_v2_cadence5_predonly_top2M)
#   3: A_2M    / avoid_animals_v2
#   4: B_2M    / avoid_water_v2   (psf_v2_cadence5_think_predonly_top2M)
#   5: B_2M    / avoid_animals_v2
#   6: C_2M    / avoid_water_v2   (psf_v2_cadence5_grounded_predonly_top2M)
#   7: C_2M    / avoid_animals_v2

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0|1) TAG="predonly";                    INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG=""; TRACK_KEY="predonly_full" ;;
    2|3) TAG="predonly_top2M";              INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG=""; TRACK_KEY="predonly_top2M" ;;
    4|5) TAG="think_predonly_top2M";        INFER_TEMPLATE="predict_only_thinking_prompt.txt";      THINK_FLAG="--gemini-thinking-budget 512"; TRACK_KEY="think_predonly_top2M" ;;
    6|7) TAG="grounded_predonly_top2M";     INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG=""; TRACK_KEY="grounded_predonly_top2M" ;;
    *) echo "ERROR unknown array ID ${ID}" >&2; exit 2 ;;
esac

if (( ID % 2 == 0 )); then
    MODE="avoid_water_v2";   VNAME="avoid_water_v2"
else
    MODE="avoid_animals_v2"; VNAME="avoid_animals_v2"
fi

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_v2_probe"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: ${CKPT} not found" >&2; exit 1
fi

echo "=== v2 steering ID=${ID}  TRACK=${TRACK_KEY}  MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/${INFER_TEMPLATE}" \
    ${THINK_FLAG} \
    --embedding-mode "${MODE}" --num-episodes 50 \
    --output-dir "${EVAL_BASE}/${VNAME}_50ep" \
    --wandb-name "eval_${TRACK_KEY}_${VNAME}_50ep"

echo "=== DONE ID=${ID}  TRACK=${TRACK_KEY}  MODE=${MODE} ==="
