#!/usr/bin/env bash
# Pilot: does Gemini play Craftax nontrivially when given the survive→ladder→
# upgrade→explore algorithm + a short action history? Target: return ≥ 5 (ideally 10+).
set -euo pipefail

OUT_DIR="/data/group_data/rl/geney/eval_results/gemini_play_pilot"

echo "======================================================================"
echo "Gemini plays Craftax directly — 5-ep pilot"
echo "======================================================================"
python -m llm.gemini_play \
    --num-episodes 5 \
    --max-steps 400 \
    --history-len 5 \
    --output-dir "${OUT_DIR}" \
    --save-video \
    --wandb-name gemini_play_pilot \
    --verbose

echo ""
echo "======================================================================"
echo "Results written to ${OUT_DIR}"
echo "======================================================================"
cat "${OUT_DIR}/summary.json" 2>/dev/null || true
