#!/usr/bin/env bash
# Rebuild the oracle data file with the combined (Mar + Apr-27) golden
# trajectories now consolidated under ~/Imagination/human_golden_trajs/.
#
# Pipeline (matches the v2_cadence5_predonly recipe used by C_grounded_2M):
#   1. convert_golden_trajs    -> oracle_pipeline/filtered_trajectories_combined/
#   2. gemini_label cadence=5  -> oracle_pipeline/predict_only_gemini_labels_v2_cadence5_3flash_combined/
#   3. embed predonly          -> oracle_pipeline/predict_only_embeddings_v2_cadence5_predonly_gemini_emb_combined/
#   4. merge                   -> oracle_pipeline/predict_only_final_v2_cadence5_predonly_gemini_emb_combined/
#
# All outputs go to NEW isolated dirs (per feedback_labelling_isolation.md).
#
# Cost: ~37 trajectories × ~70 cadence-5 calls/ep ≈ 2.5k calls × $0.0005 ≈
# $1.25. Walltime: 1-2h (label is the bottleneck at ~50 rpm equivalent for
# small batch).

set -euo pipefail

ORACLE_BASE="/data/group_data/rl/geney/oracle_pipeline"
FILTERED_DIR="${ORACLE_BASE}/filtered_trajectories_combined"
GEMINI_DIR="${ORACLE_BASE}/predict_only_gemini_labels_v2_cadence5_3flash_combined"
EMBED_DIR="${ORACLE_BASE}/predict_only_embeddings_v2_cadence5_predonly_gemini_emb_combined"
FINAL_DIR="${ORACLE_BASE}/predict_only_final_v2_cadence5_predonly_gemini_emb_combined"

PROMPT="/home/geney/Imagination/configs/training/templates/predict_state_only_prompt_concise.txt"

echo "=== Phase 1: convert golden trajectories from ~/Imagination/human_golden_trajs/ ==="
PYTHONPATH=/home/geney/Imagination python -m pipeline.convert_golden_trajs \
    --output-dir "${FILTERED_DIR}"

echo
echo "=== Phase 2: gemini_label (predict-only, cadence=5, 3-flash) ==="
PYTHONPATH=/home/geney/Imagination python -m pipeline.gemini_label \
    --filtered-dir "${FILTERED_DIR}" \
    --output-dir "${GEMINI_DIR}" \
    --gemini-model gemini-3-flash-preview \
    --thinking-budget 0 \
    --predict-only \
    --template-path "${PROMPT}"

echo
echo "=== Phase 3: embed (predonly extraction, gemini-embedding-001 3072-d) ==="
PYTHONPATH=/home/geney/Imagination python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend gemini_embed \
    --output-dim 3072 \
    --batch-size 16 \
    --extract-prediction-only

echo
echo "=== Phase 4: merge ==="
PYTHONPATH=/home/geney/Imagination python -m pipeline.merge \
    --filtered-dir "${FILTERED_DIR}" \
    --gemini-dir "${GEMINI_DIR}" \
    --embed-dir "${EMBED_DIR}" \
    --output-dir "${FINAL_DIR}"

echo
echo "=== DONE ==="
ls -la "${FINAL_DIR}/"
