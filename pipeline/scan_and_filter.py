#!/usr/bin/env python3
"""
Step 1+2: Scan all shards, compute episode returns, filter, and build an index.

Reads each shard's tar.gz, extracts NPZ files to WORK_DIR, computes per-episode
returns using the same logic as awr_llm_augmented.py, filters by MIN_EPISODE_RETURN,
and writes an index JSON listing every surviving sample and which Gemini label
it should use.

Usage:
    python -m pipeline.scan_and_filter [--min-return 15] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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


def compute_return_to_go(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    num_envs: int,
) -> np.ndarray:
    """Compute return-to-go, handling interleaved multi-env format."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    assert rewards.shape == dones.shape

    if num_envs > 1 and rewards.shape[0] % num_envs == 0:
        R = rewards.reshape(-1, num_envs)
        D = dones.reshape(-1, num_envs)
        rtg = np.zeros_like(R, dtype=np.float32)
        nxt = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(R.shape[0])):
            nxt = R[t] + gamma * nxt * (1.0 - D[t])
            rtg[t] = nxt
        return rtg.reshape(-1)

    rtg = np.zeros_like(rewards, dtype=np.float32)
    nxt = 0.0
    for t in reversed(range(rewards.shape[0])):
        nxt = rewards[t] + gamma * nxt * (1.0 - dones[t])
        rtg[t] = nxt
    return rtg


def find_episode_boundaries(
    dones: np.ndarray,
    num_envs: int,
) -> List[Dict]:
    """Identify episode boundaries and compute episode-level returns.

    The data is interleaved: [env0_t0, env1_t0, ..., env127_t0, env0_t1, ...].
    Episodes are per-environment streams separated by done=True.

    Returns list of dicts: {env_id, start_idx, end_idx, length}
    where start_idx/end_idx are global (flat) indices into the arrays.
    """
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    n = dones.shape[0]

    if num_envs > 1 and n % num_envs == 0:
        t_len = n // num_envs
        D = dones.reshape(t_len, num_envs)
        episodes = []
        for env in range(num_envs):
            ep_start_t = 0  # timestep within this env's stream
            for t in range(t_len):
                if D[t, env] > 0.5:
                    # Episode from ep_start_t to t (inclusive) for this env
                    episodes.append({
                        "env_id": env,
                        "start_t": ep_start_t,
                        "end_t": t,
                        "length": t - ep_start_t + 1,
                    })
                    ep_start_t = t + 1
            # Trailing incomplete episode (no terminal done)
            if ep_start_t < t_len:
                episodes.append({
                    "env_id": env,
                    "start_t": ep_start_t,
                    "end_t": t_len - 1,
                    "length": t_len - ep_start_t,
                    "truncated": True,
                })
        return episodes

    # Fallback: single stream
    episodes = []
    ep_start = 0
    for i in range(n):
        if dones[i] > 0.5:
            episodes.append({
                "env_id": 0,
                "start_t": ep_start,
                "end_t": i,
                "length": i - ep_start + 1,
            })
            ep_start = i + 1
    if ep_start < n:
        episodes.append({
            "env_id": 0,
            "start_t": ep_start,
            "end_t": n - 1,
            "length": n - ep_start,
            "truncated": True,
        })
    return episodes


def extract_shards(shard_dir: Path, work_dir: Path) -> List[Path]:
    """Extract all shard tar.gz files to work_dir. Returns list of NPZ paths."""
    work_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(shard_dir.glob("shard_*.tar.gz"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.tar.gz files found in {shard_dir}")

    print(f"Found {len(shard_files)} shard archives.")
    npz_paths = []

    for i, shard_path in enumerate(shard_files):
        # Check if already extracted by looking at manifest
        manifest_path = shard_path.with_suffix("").with_suffix(".files")
        if manifest_path.exists():
            manifest_lines = manifest_path.read_text().strip().splitlines()
            all_exist = all(
                (work_dir / line.strip()).exists() or (work_dir.parent / line.strip()).exists()
                for line in manifest_lines if line.strip()
            )
            if all_exist:
                for line in manifest_lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Manifests use relative paths like "new_craftax_llm.../traj...npz"
                    candidate = work_dir.parent / line
                    if candidate.exists():
                        npz_paths.append(candidate)
                    elif (work_dir / Path(line).name).exists():
                        npz_paths.append(work_dir / Path(line).name)
                continue

        print(f"  Extracting shard {i+1}/{len(shard_files)}: {shard_path.name}...", end="", flush=True)
        with tarfile.open(shard_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".npz"):
                    # Extract to work_dir with flat name
                    member.name = Path(member.name).name
                    tar.extract(member, path=work_dir)
                    npz_paths.append(work_dir / member.name)
        print(" done")

    npz_paths = sorted(set(npz_paths))
    print(f"Total NPZ files available: {len(npz_paths)}")
    return npz_paths


def scan_npz(
    npz_path: Path,
    num_envs: int,
    gamma: float,
    min_return: float,
) -> Dict:
    """Scan a single NPZ file. Returns metadata + surviving sample info."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  ERROR loading {npz_path.name}: {e}")
        return {"path": str(npz_path), "error": str(e), "episodes": [], "n_samples": 0}

    rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    dones = np.asarray(data["done"], dtype=np.float32).reshape(-1)
    n_samples = rewards.shape[0]

    # Compute return-to-go
    rtg = compute_return_to_go(rewards, dones, gamma, num_envs)

    # Find episode boundaries
    episodes = find_episode_boundaries(dones, num_envs)

    # Compute per-episode total return (= rtg at episode start)
    t_len = n_samples // num_envs if num_envs > 1 and n_samples % num_envs == 0 else n_samples
    is_interleaved = num_envs > 1 and n_samples % num_envs == 0

    episode_returns = []
    surviving_episodes = []
    all_returns = []

    for ep in episodes:
        if is_interleaved:
            # Global index for start of this episode's first step
            start_global = ep["start_t"] * num_envs + ep["env_id"]
            ep_return = float(rtg[start_global])
        else:
            ep_return = float(rtg[ep["start_t"]])

        all_returns.append(ep_return)
        ep["episode_return"] = ep_return

        if ep_return >= min_return and not ep.get("truncated", False):
            surviving_episodes.append(ep)

    # For surviving episodes, identify which timesteps need Gemini labels
    # Gemini called every GEMINI_STEP_CADENCE steps within each env's episode
    gemini_calls = []
    surviving_sample_count = 0

    for ep in surviving_episodes:
        env_id = ep["env_id"]
        for env_t in range(ep["start_t"], ep["end_t"] + 1):
            surviving_sample_count += 1
            # Relative step within episode
            ep_step = env_t - ep["start_t"]
            if ep_step % GEMINI_STEP_CADENCE == 0:
                # This timestep gets a Gemini label
                # It needs obs from env_t to min(env_t + GEMINI_STEP_CADENCE, ep end)
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
        "path": str(npz_path),
        "n_samples": n_samples,
        "n_episodes": len(episodes),
        "n_surviving_episodes": len(surviving_episodes),
        "n_surviving_samples": surviving_sample_count,
        "n_gemini_calls": len(gemini_calls),
        "return_stats": {
            "min": float(np.min(all_returns)) if all_returns else 0,
            "max": float(np.max(all_returns)) if all_returns else 0,
            "mean": float(np.mean(all_returns)) if all_returns else 0,
            "median": float(np.median(all_returns)) if all_returns else 0,
        },
        "all_returns": [float(r) for r in all_returns],
        "surviving_episodes": surviving_episodes,
        "gemini_calls": gemini_calls,
    }


def print_return_histogram(all_returns: List[float], min_return: float):
    """Print ASCII histogram of episode returns."""
    if not all_returns:
        print("  No episodes found.")
        return

    arr = np.array(all_returns)
    bins = np.arange(0, max(arr.max() + 2, min_return + 5), 2)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if len(counts) > 0 else 1
    bar_width = 50

    print(f"\n  Episode Return Distribution ({len(arr)} episodes)")
    print(f"  {'Range':>12s}  {'Count':>6s}  Bar")
    print(f"  {'─' * 12}  {'─' * 6}  {'─' * bar_width}")
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        marker = " ◄── threshold" if lo <= min_return < hi else ""
        bar = "█" * bar_len
        print(f"  [{lo:5.1f},{hi:5.1f})  {counts[i]:6d}  {bar}{marker}")

    print(f"\n  Stats: min={arr.min():.2f} max={arr.max():.2f} "
          f"mean={arr.mean():.2f} median={np.median(arr):.2f}")
    keep = (arr >= min_return).sum()
    print(f"  Filter (>= {min_return}): keeping {keep}/{len(arr)} episodes "
          f"({100*keep/len(arr):.1f}%)")


def run(
    min_return: float = MIN_EPISODE_RETURN,
    dry_run: bool = False,
    max_shards: int = None,
):
    """Main scan-and-filter entry point."""
    print("=" * 70)
    print("STEP 1+2: Scan shards, compute returns, filter episodes")
    print("=" * 70)
    print(f"  Shard directory: {SHARD_DIR}")
    print(f"  Work directory:  {WORK_DIR}")
    print(f"  Min episode return: {min_return}")
    print(f"  Gamma: {GAMMA}, Num envs: {NUM_ENVS}")
    print(f"  Gemini cadence: every {GEMINI_STEP_CADENCE} steps")
    print()

    # Check for existing extracted data
    existing_npzs = sorted(WORK_DIR.glob("trajectories_batch_*.npz")) if WORK_DIR.exists() else []
    # Also check the original source directory
    orig_dir = SHARD_DIR.parent / "new_craftax_llm_labelled_results"
    if orig_dir.exists():
        existing_npzs.extend(sorted(orig_dir.glob("trajectories_batch_*.npz")))

    if existing_npzs:
        print(f"Found {len(existing_npzs)} pre-existing NPZ files (skipping extraction).")
        npz_paths = sorted(set(existing_npzs))
    else:
        npz_paths = extract_shards(SHARD_DIR, WORK_DIR)

    if max_shards:
        npz_paths = npz_paths[:max_shards]

    # Scan each NPZ
    print(f"\nScanning {len(npz_paths)} NPZ files...")
    all_results = []
    all_returns = []
    total_samples = 0
    total_surviving = 0
    total_gemini = 0
    t0 = time.time()

    for i, npz_path in enumerate(npz_paths):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(npz_paths) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(npz_paths)}] {npz_path.name} "
                  f"({rate:.1f} files/s, ETA {eta:.0f}s)")

        result = scan_npz(npz_path, NUM_ENVS, GAMMA, min_return)
        all_results.append(result)
        all_returns.extend(result.get("all_returns", []))
        total_samples += result["n_samples"]
        total_surviving += result["n_surviving_samples"]
        total_gemini += result["n_gemini_calls"]

    elapsed = time.time() - t0
    print(f"\nScan complete in {elapsed:.1f}s")

    # Print histogram
    print_return_histogram(all_returns, min_return)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total NPZ files:        {len(npz_paths)}")
    print(f"  Total samples:           {total_samples:,}")
    print(f"  Total episodes:          {len(all_returns):,}")
    print(f"  Surviving samples:       {total_surviving:,} "
          f"({100*total_surviving/max(1,total_samples):.1f}%)")
    print(f"  Gemini API calls needed: {total_gemini:,}")
    print()

    # Gemini cost estimate
    avg_input_tokens = 2500  # ~2-3K tokens per prompt
    avg_output_tokens = 300  # ~200-400 tokens per response
    input_cost_per_m = 0.15  # Gemini 2.5 Flash
    output_cost_per_m = 0.60
    est_input_cost = total_gemini * avg_input_tokens / 1e6 * input_cost_per_m
    est_output_cost = total_gemini * avg_output_tokens / 1e6 * output_cost_per_m
    est_total_cost = est_input_cost + est_output_cost

    print(f"  Gemini cost estimate:")
    print(f"    Input:  {total_gemini * avg_input_tokens / 1e6:.1f}M tokens × "
          f"${input_cost_per_m}/M = ${est_input_cost:.2f}")
    print(f"    Output: {total_gemini * avg_output_tokens / 1e6:.1f}M tokens × "
          f"${output_cost_per_m}/M = ${est_output_cost:.2f}")
    print(f"    Total:  ~${est_total_cost:.2f}")
    print()

    if dry_run:
        print("DRY RUN — not saving index.")
        return all_results

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
            "n_files": len(npz_paths),
            "n_samples": total_samples,
            "n_episodes": len(all_returns),
            "n_surviving_samples": total_surviving,
            "n_gemini_calls": total_gemini,
            "estimated_gemini_cost_usd": round(est_total_cost, 2),
        },
        "files": [
            {
                "path": r["path"],
                "n_samples": r["n_samples"],
                "n_surviving_samples": r["n_surviving_samples"],
                "n_gemini_calls": r["n_gemini_calls"],
                "surviving_episodes": r["surviving_episodes"],
                "gemini_calls": r["gemini_calls"],
            }
            for r in all_results
            if r["n_gemini_calls"] > 0
        ],
    }

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index saved to {INDEX_PATH}")
    print(f"  ({len(index['files'])} files with surviving data)")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Scan shards and filter by episode return")
    parser.add_argument("--min-return", type=float, default=MIN_EPISODE_RETURN,
                        help=f"Minimum episode return to keep (default: {MIN_EPISODE_RETURN})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't save index, just print stats")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Process only first N NPZ files (for testing)")
    args = parser.parse_args()
    run(min_return=args.min_return, dry_run=args.dry_run, max_shards=args.max_shards)


if __name__ == "__main__":
    main()
