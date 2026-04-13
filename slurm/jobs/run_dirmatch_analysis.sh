#!/usr/bin/env bash
set -euo pipefail
python -m analysis.direction_match \
    --eval-root /data/group_data/rl/geney/eval_results \
    --policies awr_aug_debug_dirmatch freeze_obs_bcawr_dirmatch awr_bc_aug_debug_dirmatch \
    --dump-json /data/group_data/rl/geney/eval_results/direction_match_summary.json
