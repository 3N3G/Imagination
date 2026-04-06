#!/usr/bin/env bash
# BC+AWR v6 sweep — PSF-consistent oracle data, top v5 configs.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job sweep_v6 \
    --gpu L40S \
    --mem 128G \
    --time 12:00:00 \
    --array 0-5 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/sweep_v6_run.sh"
