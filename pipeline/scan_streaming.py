#!/usr/bin/env python3
"""
Step 1 (streaming): Scan all shards with minimal disk usage.

Extracts one NPZ at a time to a temp file, reads only reward+done arrays
(lazy np.load), then deletes the temp file. Peak disk usage ~10GB per NPZ.
Produces the same index JSON as scan_and_filter.py.

Usage:
    python -m pipeline.scan_streaming [--min-return 15] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.config import (
    GAMMA,
    GEMINI_STEP_CADENCE,
    INDEX_PATH,
    MIN_EPISODE_RETURN,
    NUM_ENVS,
    SHARD_DIR,
    WORK_DIR,
)
from pipeline.scan_and_filter import (
    compute_return_to_go,
    find_episode_boundaries,
    print_return_histogram,
)


def scan_npz_file(
    npz_path: str,
    npz_name: str,
    num_envs: int,
    gamma: float,
    min_return: float,
) -> Dict:
    """Scan a single NPZ file on disk. Uses lazy loading — only reads reward+done."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  ERROR loading {npz_name}: {e}")
        return {"name": npz_name, "error": str(e),
                "n_samples": 0, "all_returns": [], "surviving_episodes": [],
                "gemini_calls": [], "n_surviving_samples": 0, "n_gemini_calls": 0}

    rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["done"], dtype=np.float32).reshape(-1)
    n_samples = rewards.shape[0]

    rtg = compute_return_to_go(rewards, dones, gamma, num_envs)
    episodes = find_episode_boundaries(dones, num_envs)

    is_interleaved = num_envs > 1 and n_samples % num_envs == 0
    all_returns = []
    surviving_episodes = []

    for ep in episodes:
        if is_interleaved:
            start_global = ep["start_t"] * num_envs + ep["env_id"]
            ep_return = float(rtg[start_global])
        else:
            ep_return = float(rtg[ep["start_t"]])
        all_returns.append(ep_return)
        ep["episode_return"] = ep_return
        if ep_return >= min_return and not ep.get("truncated", False):
            surviving_episodes.append(ep)

    gemini_calls = []
    surviving_sample_count = 0
    for ep in surviving_episodes:
        env_id = ep["env_id"]
        for env_t in range(ep["start_t"], ep["end_t"] + 1):
            surviving_sample_count += 1
            ep_step = env_t - ep["start_t"]
            if ep_step % GEMINI_STEP_CADENCE == 0:
                future_end = min(env_t + GEMINI_STEP_CADENCE, ep["end_t"])
                if is_interleaved:
                    global_idx = env_t * num_envs + env_id
                else:
                    global_idx = env_t
                gemini_calls.append({
                    "env_id": env_id,
                    "env_t": env_t,
                    "global_idx": global_idx,
                    "future_end_t": future_end,
                    "n_future_steps": future_end - env_t,
                })

    data.close()

    return {
        "name": npz_name,
        "n_samples": n_samples,
        "n_episodes": len(episodes),
        "n_surviving_episodes": len(surviving_episodes),
        "n_surviving_samples": surviving_sample_count,
        "n_gemini_calls": len(gemini_calls),
        "all_returns": all_returns,
        "surviving_episodes": surviving_episodes,
        "gemini_calls": gemini_calls,
    }


def run(
    min_return: float = MIN_EPISODE_RETURN,
    dry_run: bool = False,
    max_shards: int | None = None,
):
    """Streaming scan — extracts one NPZ at a time to temp, reads reward+done, deletes."""
    print("=" * 70)
    print("STEP 1 (streaming): Scan shards, compute returns, filter episodes")
    print("=" * 70)
    print(f"  Shard directory: {SHARD_DIR}")
    print(f"  Min episode return: {min_return}")
    print(f"  Gamma: {GAMMA}, Num envs: {NUM_ENVS}")
    print(f"  Gemini cadence: every {GEMINI_STEP_CADENCE} steps")
    print()

    shard_files = sorted(SHARD_DIR.glob("shard_*.tar.gz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.tar.gz in {SHARD_DIR}")

    if max_shards:
        shard_files = shard_files[:max_shards]

    print(f"Processing {len(shard_files)} shard archives (one NPZ at a time)...")

    # Use a persistent temp dir so we can reuse across NPZs within a shard
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    work_tmp = os.path.join(tmpdir, "pipeline_scan_tmp")
    os.makedirs(work_tmp, exist_ok=True)

    all_file_results = []
    all_returns = []
    total_samples = 0
    total_surviving = 0
    total_gemini = 0
    t0 = time.time()

    for si, shard_path in enumerate(shard_files):
        elapsed = time.time() - t0
        rate = (si + 1) / elapsed if elapsed > 0 else 0
        eta = (len(shard_files) - si) / rate if rate > 0 else 0

        if si % 5 == 0 or si == len(shard_files) - 1:
            print(f"  Shard [{si+1}/{len(shard_files)}] {shard_path.name}  "
                  f"({rate:.2f} shards/s, ETA {eta/60:.1f}min)")

        try:
            with tarfile.open(shard_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".npz"):
                        continue
                    npz_name = Path(member.name).name
                    # Extract to temp file
                    tmp_path = os.path.join(work_tmp, npz_name)
                    try:
                        member.name = npz_name  # flatten path
                        tar.extract(member, path=work_tmp)

                        result = scan_npz_file(
                            tmp_path, npz_name, NUM_ENVS, GAMMA, min_return
                        )
                        all_file_results.append(result)
                        all_returns.extend(result.get("all_returns", []))
                        total_samples += result["n_samples"]
                        total_surviving += result["n_surviving_samples"]
                        total_gemini += result["n_gemini_calls"]
                    finally:
                        # Always clean up temp file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        except Exception as e:
            print(f"  ERROR with shard {shard_path.name}: {e}")

    # Clean up temp dir
    try:
        os.rmdir(work_tmp)
    except OSError:
        pass

    elapsed = time.time() - t0
    print(f"\nScan complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Histogram
    print_return_histogram(all_returns, min_return)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total NPZ files:        {len(all_file_results)}")
    print(f"  Total samples:           {total_samples:,}")
    print(f"  Total episodes:          {len(all_returns):,}")
    print(f"  Surviving samples:       {total_surviving:,} "
          f"({100*total_surviving/max(1,total_samples):.1f}%)")
    print(f"  Gemini API calls needed: {total_gemini:,}")
    print()

    # Cost estimate
    avg_input_tokens = 2500
    avg_output_tokens = 300
    input_cost_per_m = 0.15
    output_cost_per_m = 0.60
    est_input_cost = total_gemini * avg_input_tokens / 1e6 * input_cost_per_m
    est_output_cost = total_gemini * avg_output_tokens / 1e6 * output_cost_per_m
    est_total_cost = est_input_cost + est_output_cost

    print(f"  Gemini cost estimate:")
    print(f"    Input:  {total_gemini * avg_input_tokens / 1e6:.1f}M tokens x "
          f"${input_cost_per_m}/M = ${est_input_cost:.2f}")
    print(f"    Output: {total_gemini * avg_output_tokens / 1e6:.1f}M tokens x "
          f"${output_cost_per_m}/M = ${est_output_cost:.2f}")
    print(f"    Total:  ~${est_total_cost:.2f}")
    print()

    if dry_run:
        print("DRY RUN — not saving index.")
        return all_file_results

    # Save index
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    index = {
        "config": {
            "min_return": min_return,
            "gamma": GAMMA,
            "num_envs": NUM_ENVS,
            "gemini_cadence": GEMINI_STEP_CADENCE,
        },
        "summary": {
            "n_files": len(all_file_results),
            "n_samples": total_samples,
            "n_episodes": len(all_returns),
            "n_surviving_samples": total_surviving,
            "n_gemini_calls": total_gemini,
            "estimated_gemini_cost_usd": round(est_total_cost, 2),
        },
        "files": [
            {
                "name": r["name"],
                "n_samples": r["n_samples"],
                "n_surviving_samples": r["n_surviving_samples"],
                "n_gemini_calls": r["n_gemini_calls"],
                "surviving_episodes": r["surviving_episodes"],
                "gemini_calls": r["gemini_calls"],
            }
            for r in all_file_results
            if r["n_gemini_calls"] > 0
        ],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index saved to {INDEX_PATH}")
    print(f"  ({len(index['files'])} files with surviving data)")

    return all_file_results


def main():
    parser = argparse.ArgumentParser(
        description="Streaming scan of shards (minimal disk usage)")
    parser.add_argument("--min-return", type=float, default=MIN_EPISODE_RETURN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()
    run(min_return=args.min_return, dry_run=args.dry_run, max_shards=args.max_shards)


if __name__ == "__main__":
    main()
