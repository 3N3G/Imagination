#!/usr/bin/env bash
# Re-run PPO-RNN 1e8 on Craftax-Symbolic-v1 with policy checkpointing
# enabled, dropping the orbax checkpoint into a known location so we can
# load it for eval + as a base for online→BC experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTEPS="${1:-1e8}"
shift 2>/dev/null || true

SAVE_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_1e8_baseline"
mkdir -p "${SAVE_DIR}"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax \
    --job ppo_rnn_1e8_save \
    --gpu L40S \
    --mem 32G \
    --time 18:00:00 \
    "$@" \
    -- python -m online_rl.ppo_rnn \
        --env_name Craftax-Symbolic-v1 \
        --total_timesteps "$TIMESTEPS" \
        --save_policy \
        --save_output_dir "${SAVE_DIR}"
