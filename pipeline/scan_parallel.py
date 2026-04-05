#!/usr/bin/env python3
"""
Step 1 (parallel): Scan all shards using multiple CPUs.

Strategy:
  1. Scan pre-extracted NPZs in pipeline_workdir/ directly (no decompression).
  2. For remaining NPZs still in tar.gz shards, extract one-at-a-time to /tmp
     per worker, scan, delete.
  3. Merge results, save episode_index.json.

Uses ProcessPoolExecutor for parallelism. Each worker is independent.

Usage:
    python -u -m pipeline.scan_parallel [--min-return 15] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Per-NPZ scan (same logic as scan_streaming.scan_npz_file)
# ---------------------------------------------------------------------------

def scan_npz_file(
    npz_path: str,
    npz_name: str,
    num_envs: int,
    gamma: float,
    min_return: float,
    gemini_cadence: int,
) -> Dict:
    """Scan a single NPZ. Returns metadata + surviving episodes + gemini calls."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return {"name": npz_name, "path": npz_path, "error": str(e),
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
            if ep_step % gemini_cadence == 0:
                future_end = min(env_t + gemini_cadence, ep["end_t"])
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
        "path": npz_path,
        "n_samples": n_samples,
        "n_episodes": len(episodes),
        "n_surviving_episodes": len(surviving_episodes),
        "n_surviving_samples": surviving_sample_count,
        "n_gemini_calls": len(gemini_calls),
        "all_returns": all_returns,
        "surviving_episodes": surviving_episodes,
        "gemini_calls": gemini_calls,
    }


# ---------------------------------------------------------------------------
# Worker: scan a pre-extracted NPZ on disk
# ---------------------------------------------------------------------------

def worker_scan_disk(args: tuple) -> Dict:
    """Worker for scanning an already-extracted NPZ file."""
    npz_path, npz_name, num_envs, gamma, min_return, cadence = args
    return scan_npz_file(npz_path, npz_name, num_envs, gamma, min_return, cadence)


# ---------------------------------------------------------------------------
# Worker: extract + scan NPZs from one shard, then clean up
# ---------------------------------------------------------------------------

def worker_scan_shard(args: tuple) -> List[Dict]:
    """Worker for one shard: extract each NPZ to /tmp, scan, delete."""
    shard_path, already_have, num_envs, gamma, min_return, cadence, worker_id = args
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    work_tmp = os.path.join(tmpdir, f"scan_worker_{worker_id}_{os.getpid()}")
    os.makedirs(work_tmp, exist_ok=True)

    results = []
    try:
        with tarfile.open(shard_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".npz"):
                    continue
                npz_name = Path(member.name).name
                if npz_name in already_have:
                    continue  # already scanned from disk

                tmp_path = os.path.join(work_tmp, npz_name)
                try:
                    member.name = npz_name  # flatten path
                    tar.extract(member, path=work_tmp)
                    result = scan_npz_file(
                        tmp_path, npz_name, num_envs, gamma, min_return, cadence,
                    )
                    results.append(result)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    except Exception as e:
        print(f"  ERROR processing shard {Path(shard_path).name}: {e}", flush=True)

    # Clean up temp dir
    try:
        os.rmdir(work_tmp)
    except OSError:
        pass

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    min_return: float = MIN_EPISODE_RETURN,
    dry_run: bool = False,
    max_workers: int = 16,
):
    print("=" * 70)
    print("STEP 1 (parallel): Scan shards, compute returns, filter episodes")
    print("=" * 70)
    print(f"  Shard directory: {SHARD_DIR}")
    print(f"  Min episode return: {min_return}")
    print(f"  Gamma: {GAMMA}, Num envs: {NUM_ENVS}")
    print(f"  Gemini cadence: every {GEMINI_STEP_CADENCE} steps")
    print(f"  Workers: {max_workers}")
    print(flush=True)

    # Phase 1: Discover pre-extracted NPZs
    disk_npzs = sorted(WORK_DIR.glob("trajectories_batch_*.npz")) if WORK_DIR.exists() else []
    disk_npz_names = {p.name for p in disk_npzs}
    print(f"  Pre-extracted NPZs in workdir: {len(disk_npzs)}")

    # Phase 2: Discover shards
    shard_files = sorted(SHARD_DIR.glob("shard_*.tar.gz"))
    print(f"  Shard archives: {len(shard_files)}")
    print(flush=True)

    all_results = []
    all_returns = []
    total_samples = 0
    total_surviving = 0
    total_gemini = 0
    t0 = time.time()

    # Phase 1: Scan pre-extracted NPZs in parallel
    if disk_npzs:
        print(f"\n--- Phase 1: Scanning {len(disk_npzs)} pre-extracted NPZs ---", flush=True)
        disk_args = [
            (str(p), p.name, NUM_ENVS, GAMMA, min_return, GEMINI_STEP_CADENCE)
            for p in disk_npzs
        ]
        done_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker_scan_disk, a): a[1] for a in disk_args}
            for fut in as_completed(futures):
                done_count += 1
                result = fut.result()
                all_results.append(result)
                all_returns.extend(result.get("all_returns", []))
                total_samples += result["n_samples"]
                total_surviving += result["n_surviving_samples"]
                total_gemini += result["n_gemini_calls"]
                if done_count % 10 == 0 or done_count == len(disk_npzs):
                    elapsed = time.time() - t0
                    print(f"  Phase 1: [{done_count}/{len(disk_npzs)}] "
                          f"({elapsed:.0f}s elapsed)", flush=True)

        print(f"  Phase 1 complete: {len(disk_npzs)} NPZs scanned in "
              f"{time.time() - t0:.0f}s", flush=True)

    # Phase 2: Stream-scan remaining NPZs from shards in parallel
    if shard_files:
        print(f"\n--- Phase 2: Streaming scan of {len(shard_files)} shards "
              f"(skipping {len(disk_npz_names)} already-scanned) ---", flush=True)
        t1 = time.time()

        shard_args = [
            (str(sp), disk_npz_names, NUM_ENVS, GAMMA, min_return,
             GEMINI_STEP_CADENCE, i)
            for i, sp in enumerate(shard_files)
        ]

        done_shards = 0
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker_scan_shard, a): Path(a[0]).name
                       for a in shard_args}
            for fut in as_completed(futures):
                done_shards += 1
                shard_name = futures[fut]
                try:
                    shard_results = fut.result()
                except Exception as e:
                    print(f"  ERROR in shard {shard_name}: {e}", flush=True)
                    continue

                for result in shard_results:
                    all_results.append(result)
                    all_returns.extend(result.get("all_returns", []))
                    total_samples += result["n_samples"]
                    total_surviving += result["n_surviving_samples"]
                    total_gemini += result["n_gemini_calls"]

                elapsed = time.time() - t1
                rate = done_shards / elapsed if elapsed > 0 else 0
                eta = (len(shard_files) - done_shards) / rate if rate > 0 else 0
                if done_shards % 5 == 0 or done_shards == len(shard_files):
                    print(f"  Phase 2: [{done_shards}/{len(shard_files)}] "
                          f"shards ({rate:.2f}/s, ETA {eta/60:.1f}min) "
                          f"— {len(all_results)} total NPZs scanned", flush=True)

        print(f"  Phase 2 complete in {time.time() - t1:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal scan time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # Histogram
    print_return_histogram(all_returns, min_return)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total NPZ files:        {len(all_results)}")
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
            "n_files": len(all_results),
            "n_samples": total_samples,
            "n_episodes": len(all_returns),
            "n_surviving_samples": total_surviving,
            "n_gemini_calls": total_gemini,
            "estimated_gemini_cost_usd": round(est_total_cost, 2),
        },
        "files": [
            {
                "name": r["name"],
                "path": r["path"],
                "n_samples": r["n_samples"],
                "n_surviving_samples": r["n_surviving_samples"],
                "n_gemini_calls": r["n_gemini_calls"],
                "surviving_episodes": r["surviving_episodes"],
                "gemini_calls": r["gemini_calls"],
            }
            for r in sorted(all_results, key=lambda x: x["name"])
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
    parser = argparse.ArgumentParser(
        description="Parallel scan of shards (multi-CPU)")
    parser.add_argument("--min-return", type=float, default=MIN_EPISODE_RETURN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()
    workers = args.workers or len(os.sched_getaffinity(0))
    run(min_return=args.min_return, dry_run=args.dry_run, max_workers=workers)


if __name__ == "__main__":
    main()
