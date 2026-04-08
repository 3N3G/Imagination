#!/usr/bin/env python3
"""Select top-N episodes by return from final trajectory files (with embeddings).

Unlike pipeline/select_top_episodes.py which works on filtered data,
this works on merged final_trajectories that already have hidden_state embeddings.
Output is a single NPZ file suitable for use as oracle data in train_awr_weighted_v2.py.
"""
import argparse
import glob
import json
import os
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory with trajectories_*.npz")
    p.add_argument("--output", required=True, help="Output NPZ file path")
    p.add_argument("--top-n", type=int, default=250, help="Number of top episodes to select")
    p.add_argument("--min-episode-len", type=int, default=50, help="Min steps per episode")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "trajectories_*.npz")))
    if not files:
        raise ValueError(f"No files found in {args.data_dir}")
    print(f"Scanning {len(files)} files for episodes...")

    # First pass: find episode boundaries and returns
    episodes = []  # (file_idx, start, end, episode_return)
    for fi, fpath in enumerate(files):
        with np.load(fpath, allow_pickle=True) as data:
            done = np.asarray(data["done"]).reshape(-1)
            rtg = np.asarray(data["return_to_go"]).reshape(-1)
            reward = np.asarray(data["reward"]).reshape(-1)

            ep_start = 0
            for i in range(len(done)):
                if done[i]:
                    ep_len = i - ep_start + 1
                    ep_return = float(rtg[ep_start])  # RTG at episode start = total return
                    if ep_len >= args.min_episode_len:
                        episodes.append((fi, ep_start, i + 1, ep_return))
                    ep_start = i + 1

        if (fi + 1) % 20 == 0:
            print(f"  Scanned {fi + 1}/{len(files)} files, {len(episodes)} episodes found")

    print(f"Total episodes found: {len(episodes)}")
    if len(episodes) == 0:
        raise ValueError("No episodes found")

    # Sort by return descending, take top N
    episodes.sort(key=lambda x: x[3], reverse=True)
    selected = episodes[:args.top_n]
    print(f"\nSelected top {len(selected)} episodes:")
    print(f"  Return range: [{selected[-1][3]:.2f}, {selected[0][3]:.2f}]")
    print(f"  Mean return: {np.mean([e[3] for e in selected]):.2f}")

    # Second pass: extract selected episodes
    # Group by file for efficiency
    by_file = {}
    for fi, start, end, ret in selected:
        by_file.setdefault(fi, []).append((start, end, ret))

    all_obs_map_bits = []
    all_obs_aux = []
    all_action = []
    all_reward = []
    all_done = []
    all_rtg = []
    all_hidden = []
    obs_map_dim = None

    total_steps = 0
    for fi in sorted(by_file.keys()):
        fpath = files[fi]
        with np.load(fpath, allow_pickle=True) as data:
            if obs_map_dim is None and "obs_map_dim" in data.files:
                obs_map_dim = int(data["obs_map_dim"])

            for start, end, ret in by_file[fi]:
                all_obs_map_bits.append(np.asarray(data["obs_map_bits"][start:end]))
                all_obs_aux.append(np.asarray(data["obs_aux"][start:end]))
                all_action.append(np.asarray(data["action"][start:end]))
                all_reward.append(np.asarray(data["reward"][start:end]))
                all_done.append(np.asarray(data["done"][start:end]))
                all_rtg.append(np.asarray(data["return_to_go"][start:end]))
                if "hidden_state" in data.files:
                    all_hidden.append(np.asarray(data["hidden_state"][start:end]))
                total_steps += end - start

    print(f"  Total transitions: {total_steps:,}")

    # Concatenate and save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    save_dict = {
        "obs_map_bits": np.concatenate(all_obs_map_bits),
        "obs_aux": np.concatenate(all_obs_aux),
        "action": np.concatenate(all_action),
        "reward": np.concatenate(all_reward),
        "done": np.concatenate(all_done),
        "return_to_go": np.concatenate(all_rtg),
    }
    if obs_map_dim is not None:
        save_dict["obs_map_dim"] = np.array(obs_map_dim)
    if all_hidden:
        save_dict["hidden_state"] = np.concatenate(all_hidden)

    np.savez(args.output, **save_dict)

    # Save metadata
    meta = {
        "top_n": len(selected),
        "total_steps": total_steps,
        "return_range": [selected[-1][3], selected[0][3]],
        "mean_return": float(np.mean([e[3] for e in selected])),
        "source_dir": args.data_dir,
        "source_files": len(files),
        "total_episodes_scanned": len(episodes),
    }
    meta_path = args.output.replace(".npz", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
