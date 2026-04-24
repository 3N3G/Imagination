#!/usr/bin/env bash
# Patch-by-prompt: test whether a clearer Gemini prompt fixes B's basic-
# achievement coverage gap and C's long-tail loop collapse.
#
# Two new prompts:
#   v2_basic_coverage — adds explicit checklist of mandatory mid-game actions
#                       (place table/torch/furnace, craft both sword+pickaxe
#                        at each tier, place stone, never spam DO).
#                       Targets B's failure mode.
#   v2_long_tail      — adds explicit long-tail loop (plant saplings, sleep
#                        in safe enclosure, place torches, descend on ladder
#                        sight, make iron tools).
#                       Targets C's failure mode.
#
# Each prompt is run on its TARGET track so we can see if the deficit is
# patched. Cells:
#   0: B_thinking_2M  + v2_basic_coverage  (target failure mode)
#   1: C_grounded_2M  + v2_long_tail       (target failure mode)
#   2: B_thinking_2M  + v2_long_tail       (cross — does long-tail prompt help B?)
#   3: C_grounded_2M  + v2_basic_coverage  (cross — does basic-coverage help C?)
#   4: A_full         + v2_basic_coverage  (control — A is already good; should be ~baseline)
#   5: A_full         + v2_long_tail       (control)

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0) TAG="think_predonly_top2M";    TRACK_KEY="think_predonly_top2M";    MODE="v2_basic_coverage"; INFER_TEMPLATE="predict_only_thinking_prompt.txt";    THINK_FLAG="--gemini-thinking-budget 512" ;;
    1) TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"; MODE="v2_long_tail";      INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    2) TAG="think_predonly_top2M";    TRACK_KEY="think_predonly_top2M";    MODE="v2_long_tail";      INFER_TEMPLATE="predict_only_thinking_prompt.txt";    THINK_FLAG="--gemini-thinking-budget 512" ;;
    3) TAG="grounded_predonly_top2M"; TRACK_KEY="grounded_predonly_top2M"; MODE="v2_basic_coverage"; INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    4) TAG="predonly";                TRACK_KEY="predonly_full";           MODE="v2_basic_coverage"; INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    5) TAG="predonly";                TRACK_KEY="predonly_full";           MODE="v2_long_tail";      INFER_TEMPLATE="predict_state_only_prompt_concise.txt"; THINK_FLAG="" ;;
    *) echo "ERROR unknown cell ${ID}" >&2; exit 2 ;;
esac

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_patch_prompt"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== patch_prompt ID=${ID} TRACK=${TRACK_KEY} MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/${INFER_TEMPLATE}" \
    ${THINK_FLAG} \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_${MODE}_30ep"

echo "=== DONE ID=${ID} ==="
