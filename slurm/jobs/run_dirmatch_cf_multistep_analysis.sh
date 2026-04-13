#!/usr/bin/env bash
# Run multistep counterfactual dirmatch analyzer over the 3 policy dirs.
set -euo pipefail

EVAL_ROOT="/data/group_data/rl/geney/eval_results"
OUT="${EVAL_ROOT}/direction_match_cf_multistep_summary.json"

python -m analysis.direction_match_cf_multistep \
    --eval-root "${EVAL_ROOT}" \
    --policies \
        awr_aug_debug_dirmatch_cf_multistep \
        freeze_obs_bcawr_dirmatch_cf_multistep \
        awr_bc_aug_debug_dirmatch_cf_multistep \
    --dump-json "${OUT}"

echo "Wrote ${OUT}"
