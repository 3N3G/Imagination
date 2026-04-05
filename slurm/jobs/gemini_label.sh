#!/usr/bin/env bash
# Run Gemini labelling (CPU-only, needs GEMINI_API_KEY).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job gemini_label \
    --nogpu \
    --partition cpu \
    --mem 128G \
    --cpus 4 \
    --time 48:00:00 \
    "$@" \
    -- python -m pipeline.gemini_label
