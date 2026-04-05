#!/usr/bin/env bash
# BC+AWR v5 hyperparameter sweep — 8-job array.
# Requires re-labelled golden data (run relabel_golden.sh first).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job sweep_v5 \
    --gpu A100_80GB \
    --mem 128G \
    --time 12:00:00 \
    --array 0-15 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/sweep_v5_run.sh"
