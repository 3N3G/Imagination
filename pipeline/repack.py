#!/usr/bin/env python3
"""
Step 5: Repack NPZ files into the new compressed format.

For each original NPZ that has surviving episodes:
  1. Load original obs, action, reward, done, log_prob
  2. Filter to only surviving samples (from episodes with return >= threshold)
  3. Compress obs: bitpack map section (8217 binary dims) + float16 aux (51 dims)
  4. Add: return_to_go, hidden_state (10, 4096), text_generated, gemini_step_idx
  5. Drop: next_obs, old text_obs, old text_generated, old hidden_state
  6. Save as np.savez_compressed()

Usage:
    python -m pipeline.repack [--max-files N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from pipeline.config import (
    EMBED_HIDDEN_DIM,
    EMBED_OUTPUT_DIR,
    GAMMA,
    GEMINI_OUTPUT_DIR,
    GEMINI_STEP_CADENCE,
    INDEX_PATH,
    MAP_OBS_DIM,
    NUM_ENVS,
    REPACKED_DIR,
    WORK_DIR,
)
from pipeline.scan_and_filter import compute_return_to_go


def bitpack_obs(obs: np.ndarray) -> tuple:
    """Split observation into bitpacked map + float16 auxiliary.

    Uses the same convention as decode_obs_array() in awr_llm_augmented.py:
    first MAP_OBS_DIM (8217) dims are binary map → bitpacked, rest is float16.

    Args:
        obs: (N, 8268) float32 observation array

    Returns:
        obs_map_bits: (N, ceil(8217/8)) = (N, 1028) uint8
        obs_aux: (N, 51) float16
    """
    map_part = obs[:, :MAP_OBS_DIM]  # (N, 8217) binary
    map_uint8 = (map_part > 0.5).astype(np.uint8)
    obs_map_bits = np.packbits(map_uint8, axis=1, bitorder="little")  # (N, 1028)

    aux_part = obs[:, MAP_OBS_DIM:].astype(np.float16)  # (N, 51)
    return obs_map_bits, aux_part


def get_surviving_global_indices(
    episodes: List[Dict],
    n_samples: int,
    num_envs: int,
) -> np.ndarray:
    """Get sorted array of global (flat) indices for all surviving samples."""
    is_interleaved = num_envs > 1 and n_samples % num_envs == 0
    indices = []

    for ep in episodes:
        env_id = ep["env_id"]
        for env_t in range(ep["start_t"], ep["end_t"] + 1):
            if is_interleaved:
                global_idx = env_t * num_envs + env_id
            else:
                global_idx = env_t
            indices.append(global_idx)

    return np.array(sorted(set(indices)), dtype=np.int64)


def load_gemini_labels(npz_name: str) -> Dict[int, str]:
    """Load Gemini texts keyed by global_idx."""
    jsonl_path = GEMINI_OUTPUT_DIR / f"{npz_name}.jsonl"
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
                    labels[rec["global_idx"]] = rec["text"]
            except (json.JSONDecodeError, KeyError):
                continue
    return labels


def load_embeddings(npz_name: str) -> Dict[int, np.ndarray]:
    """Load embeddings keyed by global_idx."""
    embed_path = EMBED_OUTPUT_DIR / f"{npz_name}_embeddings.npz"
    if not embed_path.exists():
        return {}
    data = np.load(embed_path)
    indices = data["global_indices"]
    embeddings = data["embeddings"]  # (N, 10, 4096) float16
    return {int(idx): embeddings[i] for i, idx in enumerate(indices)}


def compute_gemini_step_idx(
    surviving_indices: np.ndarray,
    gemini_calls: List[Dict],
    n_samples: int,
    num_envs: int,
) -> np.ndarray:
    """For each surviving sample, find the global_idx of its Gemini label.

    Each sample uses the nearest prior Gemini label from its own environment.
    """
    is_interleaved = num_envs > 1 and n_samples % num_envs == 0

    # Build per-env sorted list of Gemini call global indices
    gemini_by_env = {}
    for call in gemini_calls:
        env_id = call["env_id"]
        if env_id not in gemini_by_env:
            gemini_by_env[env_id] = []
        gemini_by_env[env_id].append((call["env_t"], call["global_idx"]))

    for env_id in gemini_by_env:
        gemini_by_env[env_id].sort()

    result = np.full(len(surviving_indices), -1, dtype=np.int32)

    for i, global_idx in enumerate(surviving_indices):
        if is_interleaved:
            env_id = int(global_idx % num_envs)
            env_t = int(global_idx // num_envs)
        else:
            env_id = 0
            env_t = int(global_idx)

        calls = gemini_by_env.get(env_id, [])
        # Find nearest prior: largest env_t <= this env_t
        best_gemini_global = -1
        for call_env_t, call_global in calls:
            if call_env_t <= env_t:
                best_gemini_global = call_global
            else:
                break

        result[i] = best_gemini_global

    return result


def repack_npz(
    file_info: Dict,
    num_envs: int,
    gamma: float,
) -> Optional[str]:
    """Repack a single NPZ file into the new compressed format.

    Returns output path on success, None on skip/error.
    """
    npz_path = Path(file_info["path"])
    npz_name = npz_path.stem
    output_path = REPACKED_DIR / f"{npz_name}.npz"

    if output_path.exists():
        return None  # already done

    surviving_episodes = file_info["surviving_episodes"]
    gemini_calls = file_info["gemini_calls"]

    if not surviving_episodes:
        return None

    # Load original data
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  ERROR loading {npz_path}: {e}")
        return None

    obs = np.asarray(data["obs"], dtype=np.float32)
    n_samples = obs.shape[0]
    if len(obs.shape) > 2:
        obs = obs.reshape(n_samples, -1)

    action = np.asarray(data["action"]).reshape(-1)
    reward = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    done = np.asarray(data["done"], dtype=np.float32).reshape(-1)
    log_prob = np.asarray(data["log_prob"], dtype=np.float32).reshape(-1) if "log_prob" in data else np.zeros(n_samples, dtype=np.float32)
    data.close()

    # Get surviving sample indices
    surv_indices = get_surviving_global_indices(surviving_episodes, n_samples, num_envs)
    if len(surv_indices) == 0:
        return None

    # Filter arrays
    obs_f = obs[surv_indices]
    action_f = action[surv_indices].astype(np.int32)
    reward_f = reward[surv_indices]
    done_f = done[surv_indices]
    log_prob_f = log_prob[surv_indices]

    # Compute return-to-go on the FULL arrays (before filtering), then filter
    rtg_full = compute_return_to_go(reward, done, gamma, num_envs)
    rtg_f = rtg_full[surv_indices]

    # Bitpack obs
    obs_map_bits, obs_aux = bitpack_obs(obs_f)

    # Load Gemini labels and embeddings
    gemini_labels = load_gemini_labels(npz_name)
    embed_dict = load_embeddings(npz_name)

    # Compute gemini_step_idx for each surviving sample
    gemini_step_idx = compute_gemini_step_idx(
        surv_indices, gemini_calls, n_samples, num_envs,
    )

    # Build hidden_state array: (N_surviving, 4096) float16, mean-pooled
    hidden_state = np.zeros(
        (len(surv_indices), EMBED_HIDDEN_DIM),
        dtype=np.float16,
    )
    for i, sidx in enumerate(surv_indices):
        gidx = int(gemini_step_idx[i])
        if gidx in embed_dict:
            hidden_state[i] = embed_dict[gidx]

    # Build text_generated array: Gemini text for each sample's Gemini label
    text_generated = np.empty(len(surv_indices), dtype=object)
    for i in range(len(surv_indices)):
        gidx = int(gemini_step_idx[i])
        text_generated[i] = gemini_labels.get(gidx, "")

    # Save
    REPACKED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs_map_bits=obs_map_bits,
        obs_map_dim=np.int32(MAP_OBS_DIM),
        obs_aux=obs_aux,
        action=action_f,
        reward=reward_f,
        done=done_f.astype(np.uint8),
        log_prob=log_prob_f,
        return_to_go=rtg_f,
        hidden_state=hidden_state,
        text_generated=text_generated,
        gemini_step_idx=gemini_step_idx,
    )
    return str(output_path)


def run(max_files: Optional[int] = None):
    """Main repacking entry point."""
    print("=" * 70)
    print("STEP 5: Repack NPZ files in compressed format")
    print("=" * 70)
    print(f"  Output directory: {REPACKED_DIR}")
    print()

    with open(INDEX_PATH) as f:
        index = json.load(f)

    files = index["files"]
    if max_files:
        files = files[:max_files]

    print(f"  Files to repack: {len(files)}")
    print()

    t0 = time.time()
    repacked = 0
    skipped = 0
    errors = 0

    for i, file_info in enumerate(files):
        npz_name = Path(file_info["path"]).stem
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(files)}] {npz_name}...")

        result = repack_npz(file_info, NUM_ENVS, GAMMA)
        if result:
            repacked += 1
        elif result is None:
            skipped += 1
        else:
            errors += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Repacking complete in {elapsed:.1f}s")
    print(f"  Repacked: {repacked}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")

    # Print storage stats
    if REPACKED_DIR.exists():
        total_bytes = sum(f.stat().st_size for f in REPACKED_DIR.glob("*.npz"))
        print(f"  Total repacked size: {total_bytes / 1e9:.2f} GB")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Repack NPZ files")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    run(max_files=args.max_files)


if __name__ == "__main__":
    main()
