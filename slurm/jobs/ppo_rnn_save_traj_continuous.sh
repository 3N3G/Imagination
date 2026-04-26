#!/usr/bin/env bash
# PPO-RNN 1e8 with CONTINUOUS trajectory saving from update_step 1000 onwards.
# This fixes the bias in the every-10 save: with that schedule, only short
# episodes that complete WITHIN a 64-step saved batch are visible (mean
# episode return seen = 1.09 vs wandb's 27.87) — long high-return episodes
# end in the unsaved gap batches and are invisible.
#
# Continuous save from step 1000 (~65% through training, ep_return ~22 → 28)
# captures full episodes. ~525 batches × 46MB = ~24GB compressed.
# 525 × 65,536 transitions = ~34M transitions, plenty to pick top-4M from.
#
# Downstream: pipeline/filter_and_repack.py (with --num_envs 1024) +
# build_bitpacked_top_subset.py --target-rows 4000000 → gemini_label →
# embed → merge → AWR + BC+AWR (same C_grounded_2M recipe).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTEPS="${1:-1e8}"
shift 2>/dev/null || true

SAVE_DIR="/data/group_data/rl/geney/checkpoints/ppo_rnn_100M_save_traj_v2"
TRAJ_SAVE_DIR="/data/group_data/rl/geney/raw_trajectories/ppo_rnn_1e8_save_traj_continuous"
mkdir -p "${SAVE_DIR}" "${TRAJ_SAVE_DIR}"

"${SCRIPT_DIR}/submit.sh" \
    --env craftax_fast_llm \
    --job "ppo_rnn_save_continuous" \
    --gpu L40S \
    --mem 64G \
    --time 6:00:00 \
    "$@" \
    -- python -m online_rl.ppo_rnn \
        --env_name Craftax-Symbolic-v1 \
        --total_timesteps "${TIMESTEPS}" \
        --save_policy \
        --save_output_dir "${SAVE_DIR}" \
        --save_traj \
        --save_traj_every 1 \
        --save_traj_start_step 1000 \
        --traj_save_path "${TRAJ_SAVE_DIR}"
