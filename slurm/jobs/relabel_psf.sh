#!/usr/bin/env bash
# Re-label training data: predict-state-only with gemini-3.1-flash-lite-preview.
# CPU-only, long running (~9 hours for 158 files).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job relabel_psf \
    --nogpu \
    --partition cpu \
    --mem 32G \
    --cpus 4 \
    --time 24:00:00 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/relabel_psf_run.sh"
