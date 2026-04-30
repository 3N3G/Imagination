#!/usr/bin/env bash
# Full benchmark for a SCALING_C variant: performance (3 cells) + low-level
# steerability (6 cells) + high-level steerability (15 cells) = 24 cells total.
# Each cell n=30 episodes by default. Submitted as SLURM array; cells run in
# parallel, ~50-90 min wall time depending on slot availability.
#
# Usage:
#   VARIANT_TAG=<tag> NUM_EPISODES=<n> ./slurm/jobs/full_benchmark.sh
#
# Or via wrapper:
#   ./tools/run_benchmark.sh <variant_tag> [num_episodes]
#
# Output: per-cell results.json under
#   /data/user_data/geney/eval_results_temp/psf_v3_pporn_1e8_grounded_${VARIANT_TAG}_steer_score/
# Run tools/scorecard.py to generate the markdown summary.

set -euo pipefail

VARIANT_TAG="${VARIANT_TAG:-xhighb}"
NUM_EPISODES="${NUM_EPISODES:-30}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# All 24 modes (eval_online.py knows the templates for all of these except
# baseline_concise which is the no-prompt-steering case).
MODES=(
    # Performance (3)
    baseline_concise
    achievement_max_v2
    achievement_max_v2_thresh6
    # Low-level steerability — directional + intrinsic-targeted (6)
    direction_left_v2
    direction_right_v2
    direction_up_v2
    direction_down_v2
    target_drink_water_v2
    avoid_water_v2
    # High-level steerability — target / avoid / atomic-achievement (15)
    target_collect_stone_v2
    target_avoid_stone_v2
    target_place_stone_v2
    target_eat_cow_v2
    target_hunt_animals_v2
    avoid_animals_v2
    target_descend_v2
    target_stay_overworld_v2
    target_collect_sapling_v2
    target_place_plant_v2
    target_defeat_zombie_v2
    target_make_iron_pickaxe_v2
    target_collect_diamond_v2
    die_fast_v2
    survive_long_v2
)

NUM_CELLS=${#MODES[@]}
echo "=== full_benchmark.sh — variant=${VARIANT_TAG}, n=${NUM_EPISODES}, cells=${NUM_CELLS} ==="
for mode in "${MODES[@]}"; do
    echo "  - ${mode}"
done

# Submit each cell as a separate small job (parallelism via SLURM)
JOBIDS=()
for MODE in "${MODES[@]}"; do
    JOBID=$("${SCRIPT_DIR}/submit.sh" \
        --env craftax_fast_llm \
        --job "bench_${VARIANT_TAG}_${MODE}" \
        --gpu L40S \
        --mem 64G \
        --time 6:00:00 \
        -- env VARIANT_TAG="${VARIANT_TAG}" NUM_EPISODES="${NUM_EPISODES}" MODE="${MODE}" bash "${SCRIPT_DIR}/jobs/scaling_c_v_ood_eval.sh" 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
    JOBIDS+=("${JOBID}")
    echo "  submitted ${MODE}: ${JOBID}"
done

echo
echo "All ${NUM_CELLS} jobs submitted. JOBIDs: ${JOBIDS[*]}"
echo "Run scorecard with: PYTHONPATH=. python tools/scorecard.py --variant ${VARIANT_TAG}"
