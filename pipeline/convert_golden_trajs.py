#!/usr/bin/env python3
"""
Convert golden (oracle) trajectories to the filtered NPZ format expected by
the Gemini labeling / Qwen embedding / merge pipeline.

Produces a single trajectories_000000.npz in the output directory with proper
episode boundaries (done flags) so gemini_label.py can find Gemini call points.

Usage:
    python -m pipeline.convert_golden_trajs \
        --output-dir /data/group_data/rl/geney/oracle_pipeline/filtered_trajectories
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

GOLDEN_DIR = os.path.expanduser("~/Imagination/human_golden_trajs")
GAMMA = 0.99
OBS_DIM = 8268
MAP_DIM = 8217


def compute_return_to_go(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Backward cumulative discounted return."""
    rtg = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def load_single_traj(traj_dir: str) -> dict | None:
    """Load one golden trajectory, returning aligned arrays or None on failure."""
    meta_path = os.path.join(traj_dir, "metadata.json")
    obs_path = os.path.join(traj_dir, "obs_vectors.npy")
    traj_path = os.path.join(traj_dir, "trajectory.npz")

    if not all(os.path.exists(p) for p in [meta_path, obs_path, traj_path]):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    obs = np.load(obs_path, mmap_mode="r")  # (N, 8268)
    with np.load(traj_path, allow_pickle=True) as d:
        action_ids = d["action_ids"]
        rewards = d["rewards"]
        dones = d["dones"]

    n = len(obs)
    assert len(action_ids) == n and len(rewards) == n

    # Remove trailing sentinel (action_id=-1, reward=NaN) if present
    valid_mask = np.isfinite(rewards) & (action_ids >= 0)
    if not valid_mask.all():
        last_valid = np.where(valid_mask)[0]
        if len(last_valid) == 0:
            print(f"  Skipping {os.path.basename(traj_dir)}: no valid steps")
            return None
        end = last_valid[-1] + 1
        obs = np.array(obs[:end])
        action_ids = action_ids[:end]
        rewards = rewards[:end]
        dones = dones[:end]
    else:
        obs = np.array(obs)

    # Compute return-to-go per episode
    rtg = compute_return_to_go(rewards.astype(np.float32), GAMMA)

    # Build done flags: mark the last step of this trajectory as done=True
    done_arr = np.zeros(len(obs), dtype=np.uint8)
    done_arr[-1] = 1  # episode ends at the last step

    total_return = float(rewards.sum())
    print(f"  {os.path.basename(traj_dir)}: {len(obs)} steps, return={total_return:.2f}")

    return {
        "obs": obs.astype(np.float32),
        "action": action_ids.astype(np.int32),
        "reward": rewards.astype(np.float32),
        "return_to_go": rtg.astype(np.float32),
        "done": done_arr,
    }


def main():
    p = argparse.ArgumentParser(
        description="Convert golden trajectories to filtered NPZ format "
                    "for the Gemini/Qwen pipeline"
    )
    p.add_argument("--golden-dir", type=str, default=GOLDEN_DIR)
    p.add_argument("--output-dir", type=str,
                    default="/data/group_data/rl/geney/oracle_pipeline/filtered_trajectories")
    args = p.parse_args()

    print("=" * 60)
    print("Converting golden trajectories to filtered NPZ format")
    print("=" * 60)
    print(f"  Source: {args.golden_dir}")
    print(f"  Output: {args.output_dir}")
    print()

    all_obs, all_action, all_rtg, all_reward, all_done = [], [], [], [], []

    for name in sorted(os.listdir(args.golden_dir)):
        traj_dir = os.path.join(args.golden_dir, name)
        if not os.path.isdir(traj_dir) or name == "fails":
            continue

        result = load_single_traj(traj_dir)
        if result is None:
            continue

        all_obs.append(result["obs"])
        all_action.append(result["action"])
        all_rtg.append(result["return_to_go"])
        all_reward.append(result["reward"])
        all_done.append(result["done"])

    if not all_obs:
        raise ValueError("No valid golden trajectories found!")

    obs = np.concatenate(all_obs)
    action = np.concatenate(all_action)
    return_to_go = np.concatenate(all_rtg)
    reward = np.concatenate(all_reward)
    done = np.concatenate(all_done)

    print()
    print(f"Total oracle samples: {len(obs):,}")
    print(f"Total trajectories: {len(all_obs)}")
    print(f"Episode boundaries (done=1): {done.sum()}")
    print(f"Mean return: {np.mean([r.sum() for r in all_reward]):.2f}")
    print(f"Obs shape: {obs.shape}")

    # Bitpack the binary map section (matches filtered_trajectories format)
    obs_map = obs[:, :MAP_DIM].astype(np.uint8)
    obs_aux = obs[:, MAP_DIM:]
    obs_map_bits = np.packbits(obs_map, axis=1, bitorder="little")

    # Dummy log_prob (required by pipeline but not used for oracle data)
    log_prob = np.zeros(len(obs), dtype=np.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "trajectories_000000.npz")

    np.savez_compressed(
        output_path,
        obs_map_bits=obs_map_bits,
        obs_map_dim=np.int32(MAP_DIM),
        obs_aux=obs_aux.astype(np.float16),
        action=action,
        reward=reward,
        done=done,
        log_prob=log_prob,
        return_to_go=return_to_go,
    )

    file_size = os.path.getsize(output_path)
    print(f"\nSaved: {output_path} ({file_size / 1e6:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
