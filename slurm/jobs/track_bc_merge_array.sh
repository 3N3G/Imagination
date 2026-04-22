#!/usr/bin/env bash
# Track B/C: merge labels + embeddings + bitpacked into final_trajectories.
# Array:
#   0: think
#   1: grounded
set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}
case "${ID}" in
    0) TAG="think" ;;
    1) TAG="grounded" ;;
    *) echo "ERROR: unknown array ID ${ID}" >&2; exit 2 ;;
esac

BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED="${BASE}/filtered_trajectories_psf_v2_top2M"
LABELS="${BASE}/gemini_labels_psf_v2_cadence5_${TAG}_3flash"
EMB="${BASE}/embeddings_psf_v2_cadence5_${TAG}_predonly_gemini_emb"
OUT="${BASE}/final_trajectories_psf_v2_cadence5_${TAG}_predonly_gemini_emb_top2M"

mkdir -p "${OUT}"

python -m pipeline.merge \
    --filtered-dir "${FILTERED}" \
    --gemini-dir "${LABELS}" \
    --embed-dir "${EMB}" \
    --output-dir "${OUT}"

echo "=== DONE merge task ${ID} (${TAG}) ==="
