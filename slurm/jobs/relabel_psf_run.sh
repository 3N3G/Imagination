#!/usr/bin/env bash
# Re-label training data with predict-state-only embeddings (gemini-3.1-flash-lite-preview).
# Phase 4 only (Gemini labelling). Embed and merge are separate jobs.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf"

echo "=== Phase 4: Gemini re-label (predict-state-only, gemini-3.1-flash-lite-preview) ==="
echo "  Input:  ${FILTERED_DIR}"
echo "  Output: ${GEMINI_DIR}"
echo "  Max files: ${MAX_FILES:-158}"
echo ""

python -m pipeline.gemini_label \
    --filtered-dir "${FILTERED_DIR}" \
    --output-dir "${GEMINI_DIR}" \
    --gemini-model gemini-3.1-flash-lite-preview \
    --predict-only \
    --max-files "${MAX_FILES:-158}"

echo ""
echo "=== Gemini labelling complete ==="
# Quick check
echo "Files produced:"
ls -la "${GEMINI_DIR}/" | head -10
echo "Total files: $(ls "${GEMINI_DIR}"/*.jsonl 2>/dev/null | wc -l)"
