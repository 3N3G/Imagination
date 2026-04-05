#!/usr/bin/env bash
# Run Qwen3-8B embedding extraction.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job embed \
    --gpu A100_80GB \
    --mem 128G \
    --time 24:00:00 \
    "$@" \
    -- python -m pipeline.embed
