#!/usr/bin/env bash
# Embed + merge predict-state-only labels (GPU required for Qwen3-8B).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job embed_psf \
    --gpu A100_80GB \
    --mem 64G \
    --time 8:00:00 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/embed_psf_run.sh"
