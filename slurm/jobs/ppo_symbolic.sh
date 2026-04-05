#!/usr/bin/env bash
# Run PPO on Craftax-Symbolic.
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTEPS="${1:-1e8}"
shift 2>/dev/null || true

"${SCRIPT_DIR}/submit.sh" \
    --env craftax \
    --job ppo_symbolic \
    --gpu A100_80GB \
    --mem 32G \
    --time 24:00:00 \
    "$@" \
    -- python -m online_rl.ppo \
        --env_name Craftax-Symbolic-v1 \
        --total_timesteps "$TIMESTEPS"
