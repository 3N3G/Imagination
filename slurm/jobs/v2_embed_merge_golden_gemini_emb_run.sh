#!/usr/bin/env bash
# v2 embed + merge for golden (single file). Reads predict_only_gemini_labels_v2_3flash/.
set -euo pipefail

ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
FILTERED_DIR="${ORACLE_BASE}/filtered_trajectories"
GEMINI_DIR="${ORACLE_BASE}/predict_only_gemini_labels_v2_3flash"
EMBED_DIR="${ORACLE_BASE}/predict_only_embeddings_v2_gemini_emb"
FINAL_DIR="${ORACLE_BASE}/predict_only_final_v2_gemini_emb"

if [ ! -d "${GEMINI_DIR}" ]; then
    echo "ERROR: ${GEMINI_DIR} not found — run v2_relabel_golden_3flash_run.sh first." >&2
    exit 1
fi

echo "=== Phase 5: gemini_embed embedding (3072-d, golden) ==="
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend gemini_embed \
    --output-dim 3072 \
    --batch-size 16

echo ""
echo "=== Phase 6: merge (golden) ==="
python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}"

echo ""
ls -la "${FINAL_DIR}"/
