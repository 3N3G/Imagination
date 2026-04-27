#!/usr/bin/env bash
# Inner script run inside the Phase 2 SLURM job.
# Runs filter_and_repack then build_bitpacked_top_subset on the
# continuous-save PPO-RNN trajectories.
set -euo pipefail

# Output paths use /data/user_data/geney/ because the group_data quota
# is exhausted (other users immediately reconsume any freed space).
DATA_BASE="/data/user_data/geney/scaling_c_data"
mkdir -p "${DATA_BASE}"
RAW_DIR="/data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj_continuous"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories_pporn_1e8"
TOP4M_DIR="${DATA_BASE}/filtered_trajectories_psf_v3_pporn_1e8_top4M"

if [ ! -d "${RAW_DIR}" ]; then
    echo "ERROR: ${RAW_DIR} not found" >&2
    exit 1
fi

n_batches=$(ls "${RAW_DIR}"/trajectories_batch_*.npz 2>/dev/null | wc -l)
echo "Found ${n_batches} batches in ${RAW_DIR}"
if [ "${n_batches}" -lt 50 ]; then
    echo "ERROR: too few batches (${n_batches}); aborting" >&2
    exit 1
fi

echo "=== Stage 2a: filter_and_repack (min_return=15, num_envs=1024) ==="
python -m pipeline.filter_and_repack \
    --input_dir "${RAW_DIR}" \
    --output_dir "${FILTERED_DIR}" \
    --min_return 15 \
    --gamma 0.99 \
    --num_envs 1024

echo
echo "=== Stage 2b: build_bitpacked_top_subset (target 4M rows) ==="
python -m pipeline.build_bitpacked_top_subset \
    --input-dir "${FILTERED_DIR}" \
    --output-dir "${TOP4M_DIR}" \
    --target-rows 4000000

echo
echo "=== Phase 2 DONE ==="
ls -la "${TOP4M_DIR}/" | head -20
echo
echo "--- subset_manifest.json ---"
cat "${TOP4M_DIR}/subset_manifest.json"
