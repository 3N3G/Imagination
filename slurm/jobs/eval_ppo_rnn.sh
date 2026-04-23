#!/usr/bin/env bash
# Online 50-ep eval of a PPO-RNN checkpoint.
# Usage: eval_ppo_rnn.sh CKPT_LABEL [extra submit.sh args]
#   CKPT_LABEL maps to /data/group_data/rl/geney/checkpoints/ppo_rnn_<LABEL>_baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LABEL="${1:?specify checkpoint label (e.g. 5M, 20M, 100M)}"
shift 2>/dev/null || true

CKPT_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_${LABEL}_baseline"
OUT_DIR="/data/group_data/rl/geney/eval_results/ppo_rnn_${LABEL}_50ep"

if [ ! -d "${CKPT_DIR}/policies" ]; then
    echo "ERROR: ${CKPT_DIR}/policies not found" >&2; exit 1
fi

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "eval_ppo_rnn_${LABEL}" \
    --gpu L40S \
    --mem 32G \
    --time 6:00:00 \
    "$@" \
    -- env JAX_PLATFORMS=cpu python -m eval.eval_ppo_rnn \
        --checkpoint "${CKPT_DIR}" \
        --num-episodes 50 \
        --output-dir "${OUT_DIR}" \
        --wandb-name "eval_ppo_rnn_${LABEL}_50ep"
