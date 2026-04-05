#!/usr/bin/env python3
"""
Select top N episodes by return from filtered trajectories.

Scans all filtered trajectory files, identifies complete episodes,
ranks by return-to-go at episode start, and saves the top N episodes
into a new directory in the same bitpacked format.

Usage:
    python -m pipeline.select_top_episodes --top-n 250 --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.config import FILTERED_DIR


def find_episodes(done: np.ndarray):
    """Find (start, end_exclusive) for complete episodes."""
    done_indices = np.where(done)[0]
    episodes = []
    start = 0
    for di in done_indices:
        episodes.append((start, di + 1))
        start = di + 1
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Select top N episodes by return from filtered trajectories"
    )
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--samples-per-file", type=int, default=100_000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else FILTERED_DIR
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Select top {args.top_n} episodes by return")
    print("=" * 70)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Phase 1: Scan all files and collect episode metadata
    episode_meta = []  # [(file_path, ep_start, ep_end, return_at_start)]
    files = sorted(input_dir.glob("trajectories_*.npz"))
    if not files:
        print(f"ERROR: No trajectory files found in {input_dir}")
        return

    t0 = time.time()
    for fi, fpath in enumerate(files):
        data = np.load(fpath, allow_pickle=False)
        done = np.asarray(data["done"]).reshape(-1).astype(bool)
        rtg = np.asarray(data["return_to_go"]).reshape(-1).astype(np.float32)
        data.close()

        episodes = find_episodes(done)
        for ep_start, ep_end in episodes:
            episode_meta.append(
                (str(fpath), ep_start, ep_end, float(rtg[ep_start]))
            )

        if (fi + 1) % 100 == 0 or fi == 0:
            print(f"  Scanned {fi + 1}/{len(files)} files, "
                  f"{len(episode_meta)} episodes found")

    print(f"  Total episodes: {len(episode_meta)} "
          f"(scanned in {time.time() - t0:.1f}s)")

    # Phase 2: Sort by return and select top N
    episode_meta.sort(key=lambda x: x[3], reverse=True)
    selected = episode_meta[: args.top_n]

    total_steps = sum(ep_end - ep_start for _, ep_start, ep_end, _ in selected)
    print(f"\nSelected top {len(selected)} episodes:")
    print(f"  Return range: [{selected[-1][3]:.2f}, {selected[0][3]:.2f}]")
    print(f"  Total steps:  {total_steps:,}")
    print(f"  Avg length:   {total_steps / len(selected):.0f}")

    # Phase 3: Group by source file
    by_file = defaultdict(list)
    for fpath, ep_start, ep_end, ret in selected:
        by_file[fpath].append((ep_start, ep_end, ret))

    print(f"  Source files:  {len(by_file)}")

    # Phase 4: Extract and concatenate
    arrays = {
        "obs_map_bits": [],
        "obs_aux": [],
        "action": [],
        "reward": [],
        "done": [],
        "log_prob": [],
        "return_to_go": [],
    }
    obs_map_dim = None

    for fpath in sorted(by_file):
        data = np.load(fpath, allow_pickle=False)
        if obs_map_dim is None:
            obs_map_dim = int(data["obs_map_dim"])

        for ep_start, ep_end, _ in by_file[fpath]:
            s = slice(ep_start, ep_end)
            arrays["obs_map_bits"].append(np.array(data["obs_map_bits"][s]))
            arrays["obs_aux"].append(np.array(data["obs_aux"][s]))
            arrays["action"].append(np.array(data["action"][s]))
            arrays["reward"].append(np.array(data["reward"][s]))
            arrays["done"].append(np.array(data["done"][s]))
            arrays["log_prob"].append(np.array(data["log_prob"][s]))
            arrays["return_to_go"].append(np.array(data["return_to_go"][s]))
        data.close()

    for key in arrays:
        arrays[key] = np.concatenate(arrays[key], axis=0)

    total_samples = len(arrays["action"])
    print(f"\nExtracted {total_samples:,} samples total")

    # Phase 5: Save in chunks
    n_files = (total_samples + args.samples_per_file - 1) // args.samples_per_file
    for i in range(n_files):
        start = i * args.samples_per_file
        end = min(start + args.samples_per_file, total_samples)
        out_path = output_dir / f"trajectories_{i:06d}.npz"
        np.savez_compressed(
            out_path,
            obs_map_bits=arrays["obs_map_bits"][start:end],
            obs_map_dim=obs_map_dim,
            obs_aux=arrays["obs_aux"][start:end],
            action=arrays["action"][start:end],
            reward=arrays["reward"][start:end],
            done=arrays["done"][start:end],
            log_prob=arrays["log_prob"][start:end],
            return_to_go=arrays["return_to_go"][start:end],
        )
        print(f"  Saved {out_path.name}: {end - start} samples")

    # Save metadata
    meta = {
        "top_n": args.top_n,
        "total_episodes": len(selected),
        "total_steps": total_samples,
        "return_range": [float(selected[-1][3]), float(selected[0][3])],
        "n_output_files": n_files,
        "source_dir": str(input_dir),
        "source_total_episodes": len(episode_meta),
    }
    meta_path = output_dir / "selection_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    print(f"\n{'=' * 70}")
    print(f"Done! {n_files} files saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
