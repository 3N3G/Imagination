#!/usr/bin/env bash
# Train AWR on imagination-augmented data.
# Usage: ./slurm/jobs/train_awr.sh [extra submit.sh flags] [-- extra python flags]
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job train_awr \
    --gpu A100_80GB \
    --mem 64G \
    --time 12:00:00 \
    "$@" \
    -- python -m offline_rl.train_awr \
        --total-steps 100000 \
        --batch-size 256
