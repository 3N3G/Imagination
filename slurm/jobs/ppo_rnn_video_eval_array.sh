#!/usr/bin/env bash
# 50-ep eval with gameplay videos on PPO-RNN baseline policies.
#
# Cells:
#   0: 5M baseline
#   1: 20M baseline
# (1e8 checkpoint unavailable — the saving-enabled run timed out; needs
# fresh 24h+ submission to produce.)

set -euo pipefail

ID=${SLURM_ARRAY_TASK_ID}

case "${ID}" in
    0) SCALE="5M"; STEPS="5000000"; ;;
    1) SCALE="20M"; STEPS="20000000"; ;;
    *) echo "ERROR unknown cell ${ID}" >&2; exit 2 ;;
esac

CKPT_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_${SCALE}_baseline"
OUT_DIR="/data/group_data/rl/geney/eval_results/ppo_rnn_${SCALE}_50ep_video"

echo "=== PPO-RNN ${SCALE} video eval ==="
python -m eval.eval_ppo_rnn \
    --checkpoint "${CKPT_DIR}" \
    --num-episodes 30 \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_ppo_rnn_${SCALE}_video"

echo "=== DONE ${SCALE} ==="
