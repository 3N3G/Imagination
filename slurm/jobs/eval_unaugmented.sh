#!/usr/bin/env bash
# Evaluate unaugmented (obs-only) policy.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job eval_unaug \
    --gpu A100_80GB \
    --mem 32G \
    --time 2:00:00 \
    "$@" \
    -- python -m eval.eval_unaugmented
