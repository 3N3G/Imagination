#!/usr/bin/env bash
# Steerability evals on Track B (thinking template, top2M training data).
# Matches the 5 high-priority cells run on A_full and C_grounded.
#
# Cells (variant index):
#   0: target_collect_stone_v2
#   1: target_descend_v2
#   2: target_eat_cow_v2
#   3: direction_left_v2
#   4: direction_right_v2
#
# Inference uses thinking template (to match training distribution), which
# means the v2 probe target_X/direction_X will have the thinking-budget=512
# Gemini call on top of the swapped algorithm section. Our _pick_v2_template
# for these modes points the concise variant for both flavors — the result
# is concise-style target prompt being consumed by a thinking-trained
# policy. This is the OOD-at-inference confound we deprioritized initially
# but is now included for completeness per user request.

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "target_collect_stone_v2"
    "target_descend_v2"
    "target_eat_cow_v2"
    "direction_left_v2"
    "direction_right_v2"
)

MODE="${VARIANTS[$ID]}"
TAG="think_predonly_top2M"
TRACK_KEY="think_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_steer_v2"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== B_thinking steerability ID=${ID} MODE=${MODE} (MATCHED A/C PROTOCOL: concise template, no thinking budget) ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 50 \
    --output-dir "${EVAL_BASE}/${MODE}_50ep" \
    --wandb-name "eval_${TRACK_KEY}_${MODE}_50ep"

echo "=== DONE ID=${ID} MODE=${MODE} ==="
