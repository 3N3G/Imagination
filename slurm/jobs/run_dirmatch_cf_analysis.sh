#!/usr/bin/env bash
set -euo pipefail
python -m analysis.direction_match_cf \
    --eval-root /data/group_data/rl/geney/eval_results \
    --policies awr_aug_debug_dirmatch_cf freeze_obs_bcawr_dirmatch_cf awr_bc_aug_debug_dirmatch_cf \
    --dump-json /data/group_data/rl/geney/eval_results/direction_match_cf_summary.json
