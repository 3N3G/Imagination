#!/usr/bin/env bash
# 50-ep video eval on the freshly trained PPO-RNN 1e8 checkpoint
# (from job 7503746 ppo_rnn_save_traj). Produces gameplay.mp4 per episode.
set -euo pipefail
CKPT_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_100M_save_traj/policies"
OUT_DIR="/data/group_data/rl/geney/eval_results/ppo_rnn_100M_save_traj_50ep_video"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: ${CKPT_DIR} not found yet" >&2
    exit 1
fi

echo "=== PPO-RNN 1e8 (save_traj run) video eval ==="
python -m eval.eval_ppo_rnn \
    --checkpoint "${CKPT_DIR}" \
    --num-episodes 50 \
    --output-dir "${OUT_DIR}" \
    --wandb-name "eval_ppo_rnn_100M_save_traj_video"

echo "=== DONE ==="
