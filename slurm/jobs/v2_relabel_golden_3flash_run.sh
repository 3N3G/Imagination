#!/usr/bin/env bash
# v2 golden relabel — same recipe as shards but for the single-file golden
# (oracle_pipeline) data. Cost: ~2.5K calls × $0.0015 ≈ $4. Time: ~2 min.
set -euo pipefail

ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
FILTERED_DIR="${ORACLE_BASE}/filtered_trajectories"
OUTPUT_DIR="${ORACLE_BASE}/predict_only_gemini_labels_v2_3flash"

echo "=== v2 golden relabel: gemini-3-flash-preview + Decision-Tree prompt ==="
echo "  Input:  ${FILTERED_DIR}"
echo "  Output: ${OUTPUT_DIR}"

python -m pipeline.gemini_label \
    --filtered-dir "${FILTERED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --gemini-model gemini-3-flash-preview \
    --thinking-budget 0 \
    --predict-only \
    --max-files 1

echo ""
ls -la "${OUTPUT_DIR}"/
