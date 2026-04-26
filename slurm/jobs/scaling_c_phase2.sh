#!/usr/bin/env bash
# SCALING_C Phase 2 — filter + bitpack + top-K subset, on the
# continuous-save PPO-RNN 1e8 data.
#
# Inputs:
#   /data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj_continuous/
#     (~24GB compressed, 525 batches × 65536 transitions, NUM_ENVS=1024)
#
# Outputs:
#   filtered_trajectories_pporn_1e8/                (bitpacked stage, ~? GB)
#   filtered_trajectories_psf_v3_pporn_1e8_top4M/   (top-4M-rows subset)
#
# Both stages are CPU-only and modest memory (32G plenty for 24GB input).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RAW_DIR="/data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj_continuous"
SHARDS_ROOT="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${SHARDS_ROOT}/filtered_trajectories_pporn_1e8"
TOP4M_DIR="${SHARDS_ROOT}/filtered_trajectories_psf_v3_pporn_1e8_top4M"

# Optional: --dependency=afterok:JOBID can be passed via "$@"
"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "scaling_c_phase2" \
    --nogpu \
    --partition cpu \
    --mem 64G \
    --time 4:00:00 \
    "$@" \
    -- bash -c "
set -euo pipefail
echo '=== Stage 2a: filter_and_repack ==='
PYTHONPATH=. python -m pipeline.filter_and_repack \
    --input_dir '${RAW_DIR}' \
    --output_dir '${FILTERED_DIR}' \
    --min_return 15 \
    --gamma 0.99 \
    --num_envs 1024

echo
echo '=== Stage 2b: build_bitpacked_top_subset (target 4M rows) ==='
PYTHONPATH=. python -m pipeline.build_bitpacked_top_subset \
    --input-dir '${FILTERED_DIR}' \
    --output-dir '${TOP4M_DIR}' \
    --target-rows 4000000

echo
echo '=== Phase 2 DONE ==='
ls -la '${TOP4M_DIR}/'
echo
cat '${TOP4M_DIR}/subset_manifest.json'
"
