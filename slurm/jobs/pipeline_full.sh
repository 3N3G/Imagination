#!/usr/bin/env bash
# Run the full pipeline orchestrator (phases 4-6: gemini → embed → merge).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job pipeline_full \
    --gpu A100_80GB \
    --mem 128G \
    --time 48:00:00 \
    "$@" \
    -- python -m pipeline.run
