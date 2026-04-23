#!/usr/bin/env bash
# Run PPO-RNN on Craftax-Symbolic-v1 with policy checkpointing enabled,
# dropping the orbax checkpoint into a scale-specific location so we can
# load it for eval + as a base for online→BC experiments.
# Usage: ppo_rnn_save.sh [TIMESTEPS] [extra submit.sh args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTEPS="${1:-1e8}"
shift 2>/dev/null || true

# Derive a short scale label: 1e8 -> 100M, 2e7 -> 20M, 5e6 -> 5M, etc.
SCALE_LABEL=$(python3 -c "import sys; v=float(sys.argv[1]); m=v/1e6; print(f'{int(m)}M' if m>=1 and m==int(m) else (f'{m:g}M'))" "${TIMESTEPS}")
SAVE_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_${SCALE_LABEL}_baseline"
mkdir -p "${SAVE_DIR}"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax \
    --job "ppo_rnn_${SCALE_LABEL}_save" \
    --gpu L40S \
    --mem 32G \
    --time 18:00:00 \
    "$@" \
    -- python -m online_rl.ppo_rnn \
        --env_name Craftax-Symbolic-v1 \
        --total_timesteps "$TIMESTEPS" \
        --save_policy \
        --save_output_dir "${SAVE_DIR}"
