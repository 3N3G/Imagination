#!/usr/bin/env bash
# Validate v4+v5 checkpoints on re-labelled golden trajectory.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job validate_v5 \
    --gpu A100_80GB \
    --mem 32G \
    --time 2:00:00 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/validate_v5_run.sh"
