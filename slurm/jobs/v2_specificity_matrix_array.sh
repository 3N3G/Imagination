#!/usr/bin/env bash
# Specificity matrix on C_grounded_2M.
#
# Each cell tests one prompt × C_grounded. Per-episode target metrics
# (stones mined, cows eaten, drinks, monster kills, place-X action counts,
# direction-action counts) come from the replay tool — NOT just achievement
# binary rates or return.
#
# Cells (C_grounded_2M only — A_full will be added later if budget allows):
#   0: baseline (regular concise) — already exists, skip
#
#   stone axis:
#   1: target_collect_stone_v2 (more stones)
#   2: target_avoid_stone_v2   (less stones)
#   3: target_place_stone_v2   (more PLACE_STONE actions)
#
#   cow axis:
#   4: target_eat_cow_v2       (more cow eats)
#   5: target_hunt_animals_v2  (more cow eats, aggressive)
#   6: avoid_animals_v2        (less cow eats)
#
#   water axis:
#   7: target_drink_water_v2   (more drink events)
#   8: avoid_water_v2          (less drink events)
#
#   ladder axis:
#   9:  target_descend_v2       (more DESCEND)
#   10: target_stay_overworld_v2 (less DESCEND)
#
#   sapling/plant axis:
#   11: target_collect_sapling_v2 (more sapling collected)
#   12: target_place_plant_v2     (more PLACE_PLANT actions)
#
#   combat axis:
#   13: target_defeat_zombie_v2 (more zombies killed)
#
#   chain axes (rare, large headroom):
#   14: target_make_iron_pickaxe_v2 (make_iron_pickaxe achievement)
#   15: target_collect_diamond_v2   (collect_diamond achievement)
#
#   life/death axis:
#   16: die_fast_v2     (die quickly)
#   17: survive_long_v2 (max episode length)
#
#   direction axis:
#   18: direction_left_v2
#   19: direction_right_v2
#   20: direction_up_v2
#   21: direction_down_v2

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

VARIANTS=(
    "target_collect_stone_v2"     # 0
    "target_avoid_stone_v2"       # 1
    "target_place_stone_v2"       # 2
    "target_eat_cow_v2"           # 3
    "target_hunt_animals_v2"      # 4
    "avoid_animals_v2"            # 5
    "target_drink_water_v2"       # 6
    "avoid_water_v2"              # 7
    "target_descend_v2"           # 8
    "target_stay_overworld_v2"    # 9
    "target_collect_sapling_v2"   # 10
    "target_place_plant_v2"       # 11
    "target_defeat_zombie_v2"     # 12
    "target_make_iron_pickaxe_v2" # 13
    "target_collect_diamond_v2"   # 14
    "die_fast_v2"                 # 15
    "survive_long_v2"             # 16
    "direction_left_v2"           # 17
    "direction_right_v2"          # 18
    "direction_up_v2"             # 19
    "direction_down_v2"           # 20
)

MODE="${VARIANTS[$ID]}"

# All cells run on C_grounded_2M
TAG="grounded_predonly_top2M"
TRACK_KEY="grounded_predonly_top2M"

CKPT_BASE="/data/group_data/rl/geney/checkpoints/psf_v2_cadence5_${TAG}/freezenone"
EVAL_BASE="/data/group_data/rl/geney/eval_results/psf_v2_cadence5_${TRACK_KEY}_specificity"

CKPT="${CKPT_BASE}/final.pth"
STATS="${CKPT_BASE}/hidden_state_stats.npz"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ${CKPT}" >&2; exit 1; fi

echo "=== specificity ID=${ID} TRACK=${TRACK_KEY} MODE=${MODE} ==="
python -m eval.eval_online \
    --checkpoint "${CKPT}" --hidden-stats "${STATS}" \
    --embed-backend gemini_embed --hidden-dim 3072 \
    --extract-prediction-only \
    --prompt-template-path "/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt" \
    --embedding-mode "${MODE}" --num-episodes 30 \
    --output-dir "${EVAL_BASE}/${MODE}_30ep" \
    --wandb-name "eval_${TRACK_KEY}_specificity_${MODE}_30ep"

echo "=== DONE ID=${ID} ==="
