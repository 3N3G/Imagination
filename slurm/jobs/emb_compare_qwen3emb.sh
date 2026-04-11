#!/usr/bin/env bash
# Embedding comparison: Qwen3-Embedding model on PSF data (GPU).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job emb_cmp_qwen3emb \
    --gpu A100_80GB \
    --mem 256G \
    --time 24:00:00 \
    --partition rl \
    --qos rl_qos \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/emb_compare_qwen3emb_run.sh"
