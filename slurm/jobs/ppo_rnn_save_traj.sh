#!/usr/bin/env bash
# PPO-RNN 1e8 with trajectory saving for SCALING_C Phase 1.
# Saves (obs, next_obs, action, reward, done, log_prob) batches every
# 20 update_steps after the first 1500 steps (lets the policy improve
# first). At 16,384 transitions/step and ~6,103 total updates, saving
# every 20 of the last 4,603 = ~230 batches × 16,384 = ~3.8M transitions
# saved (~50GB compressed).
#
# Downstream: pipeline/scan_streaming.py + filter_and_repack.py to keep
# top 4M by RTG, then gemini_label.py + embed.py + merge.py +
# offline_rl.train_awr* — same recipe as C_grounded_2M.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTEPS="${1:-1e8}"
shift 2>/dev/null || true

SAVE_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_100M_save_traj"
TRAJ_SAVE_DIR="/data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj"
mkdir -p "${SAVE_DIR}" "${TRAJ_SAVE_DIR}"

# PPO-RNN at default config runs ~1525 update_steps for 1e8 timesteps
# (NUM_ENVS=1024 × NUM_STEPS=64 = 65,536 transitions/step, 1e8/65536 ≈ 1525).
# Saved bucket-size = save_every × 65,536. Choose save_every=10 + start=500
# → ~100 batches × 65K = ~6.5M transitions saved, covering update_steps 500
# - 1525 (ep_return ~17 → 28). Plenty to filter top-4M from.
"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "ppo_rnn_save_traj" \
    --gpu L40S \
    --mem 64G \
    --time 24:00:00 \
    "$@" \
    -- python -m online_rl.ppo_rnn \
        --env_name Craftax-Symbolic-v1 \
        --total_timesteps "${TIMESTEPS}" \
        --save_policy \
        --save_output_dir "${SAVE_DIR}" \
        --save_traj \
        --save_traj_every 10 \
        --save_traj_start_step 500 \
        --traj_save_path "${TRAJ_SAVE_DIR}"
