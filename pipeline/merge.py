#!/usr/bin/env python3
"""
Phase 6: Merge embeddings and Gemini text into filtered trajectory files.

For each filtered trajectory file, loads the corresponding Gemini JSONL and
embedding NPZ, computes gemini_step_idx for every sample (nearest prior Gemini
label within the same episode), and saves the final output with all keys.

Usage:
    python -m pipeline.merge [--max-files N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.config import (
    EMBED_HIDDEN_DIM,
    EMBED_OUTPUT_DIR,
    FILTERED_DIR,
    FINAL_DIR,
    GEMINI_OUTPUT_DIR,
    GEMINI_STEP_CADENCE,
)


def load_gemini_labels(jsonl_path: Path) -> Dict[int, str]:
    """Load Gemini texts keyed by sample_idx."""
    if not jsonl_path.exists():
        return {}
    labels = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ok") and "text" in rec:
                    labels[rec["sample_idx"]] = rec["text"]
            except (json.JSONDecodeError, KeyError):
                continue
    return labels


def load_embeddings(embed_path: Path) -> Dict[int, np.ndarray]:
    """Load embeddings keyed by sample_idx."""
    if not embed_path.exists():
        return {}
    data = np.load(embed_path)
    indices = data["sample_indices"]
    embeddings = data["embeddings"]  # (N, 4096) float16
    return {int(idx): embeddings[i] for i, idx in enumerate(indices)}


def find_episodes(done: np.ndarray) -> List[tuple]:
    """Find (start, end_exclusive) indices of complete episodes."""
    done_indices = np.where(done)[0]
    episodes = []
    start = 0
    for di in done_indices:
        episodes.append((start, di + 1))
        start = di + 1
    return episodes


def compute_gemini_step_idx(
    n_samples: int,
    episodes: List[tuple],
    gemini_labels: Dict[int, str],
    cadence: int = GEMINI_STEP_CADENCE,
) -> np.ndarray:
    """For each sample, find the sample_idx of its nearest prior Gemini label.

    Within each episode, Gemini calls happen at steps 0, cadence, 2*cadence, ...
    Each sample uses the most recent Gemini call from its own episode.
    """
    result = np.full(n_samples, -1, dtype=np.int32)

    for ep_start, ep_end in episodes:
        # Gemini call positions within this episode
        call_indices = []
        for step_in_ep in range(0, ep_end - ep_start, cadence):
            sample_idx = ep_start + step_in_ep
            if sample_idx in gemini_labels:
                call_indices.append(sample_idx)

        if not call_indices:
            continue

        # Assign each sample in the episode to its nearest prior call
        call_ptr = 0
        for idx in range(ep_start, ep_end):
            # Advance pointer if next call is at or before this index
            while (call_ptr + 1 < len(call_indices)
                   and call_indices[call_ptr + 1] <= idx):
                call_ptr += 1
            result[idx] = call_indices[call_ptr]

    return result


def merge_file(
    traj_path: Path,
    gemini_dir: Optional[Path] = None,
    embed_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Merge embeddings and text into one filtered trajectory file."""
    _gemini_dir = gemini_dir or GEMINI_OUTPUT_DIR
    _embed_dir = embed_dir or EMBED_OUTPUT_DIR
    _output_dir = output_dir or FINAL_DIR
    fname = traj_path.stem
    output_path = _output_dir / f"{fname}.npz"

    if output_path.exists():
        return None  # already done

    # Load filtered trajectory
    data = np.load(traj_path, allow_pickle=False)
    obs_map_bits = data["obs_map_bits"]
    obs_map_dim = data["obs_map_dim"]
    obs_aux = data["obs_aux"]
    action = data["action"]
    reward = data["reward"]
    done_arr = np.asarray(data["done"]).reshape(-1).astype(bool)
    log_prob = data["log_prob"]
    return_to_go = data["return_to_go"]
    data.close()

    n_samples = len(action)

    # Load Gemini labels and embeddings
    gemini_labels = load_gemini_labels(_gemini_dir / f"{fname}.jsonl")
    embed_dict = load_embeddings(_embed_dir / f"{fname}_embeddings.npz")

    if not gemini_labels:
        print(f"  WARNING: no Gemini labels for {fname}, skipping")
        return None

    # Find episodes and compute gemini_step_idx
    episodes = find_episodes(done_arr)
    gemini_step_idx = compute_gemini_step_idx(
        n_samples, episodes, gemini_labels,
    )

    # Build hidden_state array: (N, 4096) float16, mean-pooled
    hidden_state = np.zeros(
        (n_samples, EMBED_HIDDEN_DIM),
        dtype=np.float16,
    )
    missing_embeddings = 0
    for i in range(n_samples):
        gidx = int(gemini_step_idx[i])
        if gidx in embed_dict:
            hidden_state[i] = embed_dict[gidx]
        elif gidx >= 0:
            missing_embeddings += 1

    if missing_embeddings > 0:
        print(f"  WARNING: {fname} has {missing_embeddings}/{n_samples} samples "
              f"with gemini_step_idx but no embedding")

    # Build text_generated array
    text_generated = np.empty(n_samples, dtype=object)
    for i in range(n_samples):
        gidx = int(gemini_step_idx[i])
        text_generated[i] = gemini_labels.get(gidx, "")

    # Save
    _output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs_map_bits=obs_map_bits,
        obs_map_dim=obs_map_dim,
        obs_aux=obs_aux,
        action=action,
        reward=reward,
        done=done_arr.astype(np.uint8),
        log_prob=log_prob,
        return_to_go=return_to_go,
        hidden_state=hidden_state,
        text_generated=text_generated,
        gemini_step_idx=gemini_step_idx,
    )
    return str(output_path)


def run(
    max_files: Optional[int] = None,
    filtered_dir: Optional[str] = None,
    gemini_dir: Optional[str] = None,
    embed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Main merge entry point."""
    _filtered_dir = Path(filtered_dir) if filtered_dir else FILTERED_DIR
    _gemini_dir = Path(gemini_dir) if gemini_dir else GEMINI_OUTPUT_DIR
    _embed_dir = Path(embed_dir) if embed_dir else EMBED_OUTPUT_DIR
    _output_dir = Path(output_dir) if output_dir else FINAL_DIR

    print("=" * 70)
    print("PHASE 6: Merge embeddings + Gemini text into final files")
    print("=" * 70)
    print(f"  Filtered: {_filtered_dir}")
    print(f"  Gemini:   {_gemini_dir}")
    print(f"  Embed:    {_embed_dir}")
    print(f"  Output:   {_output_dir}")
    print()

    files = sorted(_filtered_dir.glob("trajectories_*.npz"))
    if not files:
        print(f"  No trajectory files found in {_filtered_dir}")
        return
    if max_files:
        files = files[:max_files]

    print(f"  Files to merge: {len(files)}")
    print()

    t0 = time.time()
    merged = 0
    skipped = 0

    for i, fpath in enumerate(files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {fpath.name}...")

        result = merge_file(fpath,
                            gemini_dir=_gemini_dir,
                            embed_dir=_embed_dir,
                            output_dir=_output_dir)
        if result:
            merged += 1
        else:
            skipped += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Merge complete in {elapsed:.1f}s")
    print(f"  Merged: {merged}")
    print(f"  Skipped: {skipped}")

    if _output_dir.exists():
        total_bytes = sum(f.stat().st_size for f in _output_dir.glob("*.npz"))
        print(f"  Total final size: {total_bytes / 1e9:.2f} GB")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Merge embeddings into trajectory files")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--filtered-dir", type=str, default=None,
                        help="Override filtered trajectories directory")
    parser.add_argument("--gemini-dir", type=str, default=None,
                        help="Override Gemini labels directory")
    parser.add_argument("--embed-dir", type=str, default=None,
                        help="Override embeddings directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override final output directory")
    args = parser.parse_args()
    run(max_files=args.max_files,
        filtered_dir=args.filtered_dir,
        gemini_dir=args.gemini_dir,
        embed_dir=args.embed_dir,
        output_dir=args.output_dir)


if __name__ == "__main__":
    main()
