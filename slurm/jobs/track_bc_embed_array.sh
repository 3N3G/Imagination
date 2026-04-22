#!/usr/bin/env bash
# Track B/C: predonly embed the label outputs with gemini-embedding-001.
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
LABELS="${BASE}/gemini_labels_psf_v2_cadence5_${TAG}_3flash"
FILTERED="${BASE}/filtered_trajectories_psf_v2_top2M"
OUT="${BASE}/embeddings_psf_v2_cadence5_${TAG}_predonly_gemini_emb"

mkdir -p "${OUT}"

python -m pipeline.embed \
    --backend gemini_embed \
    --output-dim 3072 \
    --gemini-dir "${LABELS}" \
    --output-dir "${OUT}" \
    --extract-prediction-only

echo "=== DONE embed task ${ID} (${TAG}) ==="
