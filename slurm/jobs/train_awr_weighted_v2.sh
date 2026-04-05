#!/usr/bin/env bash
# Train weighted BC+AWR v2.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job train_bcawr_v2 \
    --gpu A100_80GB \
    --mem 64G \
    --time 12:00:00 \
    "$@" \
    -- python -m offline_rl.train_awr_weighted_v2 \
        --oracle-fraction 0.10 \
        --oracle-loss-weight 2.0 \
        --entropy-coeff 0.01
