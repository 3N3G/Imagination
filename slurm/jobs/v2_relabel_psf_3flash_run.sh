#!/usr/bin/env bash
# v2 PSF relabel — gemini-3-flash-preview, new (Decision-Tree) prompt,
# fixed obs_to_text (now includes equipment), thinking disabled for cost.
# Inputs:  filtered_trajectories/*.npz (unchanged)
# Outputs: gemini_labels_psf_v2_3flash/trajectories_*.jsonl
#
# Cost estimate at 842K calls × ~$0.0015/call ≈ $1300 (rough; depends on
# 3-flash-preview pricing). Time: 842K calls / 2000 rpm ≈ 7 hours.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
FILTERED_DIR="${DATA_BASE}/filtered_trajectories"
OUTPUT_DIR="${DATA_BASE}/gemini_labels_psf_v2_3flash"

echo "=== v2 PSF relabel: gemini-3-flash-preview + Decision-Tree prompt ==="
echo "  Input:  ${FILTERED_DIR}"
echo "  Output: ${OUTPUT_DIR}"

python -m pipeline.gemini_label \
    --filtered-dir "${FILTERED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --gemini-model gemini-3-flash-preview \
    --thinking-budget 0 \
    --predict-only \
    --max-files "${MAX_FILES:-158}"

echo ""
echo "=== Done. Files written: ==="
ls -la "${OUTPUT_DIR}"/*.jsonl 2>/dev/null | wc -l
