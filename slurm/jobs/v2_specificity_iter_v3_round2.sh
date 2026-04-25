#!/usr/bin/env bash
# Round 2 of v3 prompt iteration on the remaining NULL specificity-matrix
# cells. Same pattern as the earlier avoid_stone_v3 / survive_long_v3
# iteration: rewrite to elevate the target as priority 1 (à la
# target_descend_v2), then re-run on C_grounded_2M at n=30.

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "target_eat_cow_v3"           # 0
    "target_drink_water_v3"       # 1
    "target_stay_overworld_v3"    # 2
    "target_place_plant_v3"       # 3
    "target_defeat_zombie_v3"     # 4
    "target_collect_sapling_v3"   # 5
)

MODE="${VARIANTS[$ID]}"

TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_specificity_iter"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== specificity_iter_round2 ID=${ID} TRACK=${TRACK_KEY} MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_specificity_iter_${MODE}_30ep"

echo "=== DONE ID=${ID} ==="
