#!/usr/bin/env bash
# Evaluate with full imagination pipeline (Gemini + Qwen embedding at runtime).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job eval_online \
    --gpu A100_80GB \
    --mem 64G \
    --time 6:00:00 \
    "$@" \
    -- python -m eval.eval_online
