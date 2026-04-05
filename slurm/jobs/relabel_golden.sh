#!/usr/bin/env bash
# Re-label golden trajectory with the SAME pipeline as training data:
#   gemini-2.5-flash + oracle mode (sees future 15 steps)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job relabel_golden \
    --gpu A100_80GB \
    --mem 64G \
    --time 2:00:00 \
    "$@" \
    -- bash "${SCRIPT_DIR}/jobs/relabel_golden_run.sh"
