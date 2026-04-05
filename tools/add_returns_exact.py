import argparse
import glob
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_DATA_DIR = "/data/group_data/rl/geney/vllm_craftax_labelled_results"
DEFAULT_PATTERN = "trajectories_batch_*.npz"
DEFAULT_GAMMA = 0.99


def extract_batch_index(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"trajectories_batch_(\d+)\.npz$", name)
    if m:
        return int(m.group(1))
    return -1


def sort_files(paths: List[str]) -> List[str]:
    with_idx = [(extract_batch_index(p), p) for p in paths]
    if any(idx < 0 for idx, _ in with_idx):
        return sorted(paths)
    return [p for _, p in sorted(with_idx, key=lambda x: x[0])]


def compute_rtg_with_carry(
    rewards_flat: np.ndarray,
    dones_flat: np.ndarray,
    num_envs: int,
    gamma: float,
    carry_next_return: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    rewards = np.asarray(rewards_flat, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones_flat, dtype=np.float32).reshape(-1)

    if rewards.shape[0] != dones.shape[0]:
        raise ValueError(
            f"reward/done length mismatch: {rewards.shape[0]} vs {dones.shape[0]}"
        )
    if rewards.shape[0] % num_envs != 0:
        raise ValueError(
            f"Length {rewards.shape[0]} not divisible by num_envs={num_envs}"
        )

    rewards_mat = rewards.reshape(-1, num_envs)
    dones_mat = dones.reshape(-1, num_envs)
    rtg_mat = np.zeros_like(rewards_mat, dtype=np.float32)

    next_return = carry_next_return.astype(np.float32, copy=True)
    for t in reversed(range(rewards_mat.shape[0])):
        next_return = rewards_mat[t] + gamma * next_return * (1.0 - dones_mat[t])
        rtg_mat[t] = next_return

    unfinished_at_file_end = int(np.sum(dones_mat[-1] < 0.5))
    return rtg_mat.reshape(-1), next_return, unfinished_at_file_end


def atomic_save_npz(path: str, payload: Dict[str, np.ndarray]):
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact return-to-go across trajectory batch files by carrying "
            "per-env returns across file boundaries."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--search_pattern", type=str, default=DEFAULT_PATTERN)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--start_batch", type=int, default=None)
    parser.add_argument("--end_batch", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    pattern = os.path.join(args.data_dir, args.search_pattern)
    files = sort_files(glob.glob(pattern))
    if not files:
        raise ValueError(f"No files found for pattern: {pattern}")

    if args.start_batch is not None or args.end_batch is not None:
        filtered = []
        for f in files:
            idx = extract_batch_index(f)
            if idx < 0:
                continue
            if args.start_batch is not None and idx < args.start_batch:
                continue
            if args.end_batch is not None and idx > args.end_batch:
                continue
            filtered.append(f)
        files = filtered
        if not files:
            raise ValueError("No files left after start/end batch filtering.")

    print(f"Input files: {len(files)}")
    print(f"First file: {os.path.basename(files[0])}")
    print(f"Last file:  {os.path.basename(files[-1])}")
    print(f"num_envs={args.num_envs}, gamma={args.gamma}")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")
    else:
        print("Output mode: in-place overwrite")
    if args.dry_run:
        print("Dry run enabled; no files will be written.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    total_samples = 0
    for f in files:
        with np.load(f, mmap_mode="r") as d:
            if "reward" not in d or "done" not in d:
                raise KeyError(f"{f} missing reward/done")
            n = int(d["reward"].shape[0])
            if n % args.num_envs != 0:
                raise ValueError(
                    f"{f}: reward length {n} not divisible by num_envs={args.num_envs}"
                )
            total_samples += n
    print(f"Total samples: {total_samples}")

    carry_next = np.zeros(args.num_envs, dtype=np.float32)
    unfinished_streams_total = 0

    for i, src_path in enumerate(reversed(files), start=1):
        with np.load(src_path, allow_pickle=True) as d:
            payload = {k: d[k] for k in d.files}

        rtg, carry_next, unfinished_at_end = compute_rtg_with_carry(
            payload["reward"],
            payload["done"],
            num_envs=args.num_envs,
            gamma=args.gamma,
            carry_next_return=carry_next,
        )
        payload["return_to_go"] = rtg.astype(np.float32)
        unfinished_streams_total += unfinished_at_end

        if args.output_dir:
            dst_path = os.path.join(args.output_dir, os.path.basename(src_path))
        else:
            dst_path = src_path

        if not args.dry_run:
            atomic_save_npz(dst_path, payload)

        if i % 25 == 0 or i == 1 or i == len(files):
            print(
                f"[{i}/{len(files)}] wrote {os.path.basename(dst_path)} "
                f"(unfinished_envs_at_file_end={unfinished_at_end})"
            )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"Processed files: {len(files)}")
    print(f"Processed samples: {total_samples}")
    print(
        "Total unfinished env-stream markers at file ends (expected, non-error): "
        f"{unfinished_streams_total}"
    )
    print(
        "Final carry vector norm at dataset start: "
        f"{float(np.linalg.norm(carry_next)):.6f}"
    )


if __name__ == "__main__":
    main()
