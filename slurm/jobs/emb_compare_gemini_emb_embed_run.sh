#!/usr/bin/env bash
# Phase 1 only: Gemini text-embedding-001 API embed (3072-dim) on PSF labels.
# Runs on a CPU/general partition node with internet access.
# After this completes, submit emb_compare_gemini_emb_train.sh for GPU training.
set -euo pipefail

DATA_BASE="/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards"
GEMINI_DIR="${DATA_BASE}/gemini_labels_psf"
EMBED_DIR="${DATA_BASE}/embeddings_psf_gemini_emb"
HIDDEN_DIM=3072

echo "======================================================================"
echo "PHASE 1: Gemini embedding-001 embed (PSF labels, ${HIDDEN_DIM}-dim)"
echo "======================================================================"
python -m pipeline.embed \
    --gemini-dir "${GEMINI_DIR}" \
    --output-dir "${EMBED_DIR}" \
    --backend gemini_embed \
    --output-dim "${HIDDEN_DIM}"

echo ""
echo "======================================================================"
echo "Embed complete. Submit emb_compare_gemini_emb_train.sh for GPU training."
echo "======================================================================"
