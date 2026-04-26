"""Phase 3: Filter trajectories by episode return and repack in compressed format.

Reads raw trajectory files saved by PPO (shape: NUM_STEPS, NUM_ENVS, ...),
reconstructs per-env episode timelines, filters by episode return >= threshold,
computes return_to_go, bitpacks obs, and saves in the final format compatible
with decode_obs_array() in awr_llm_augmented.py.

Usage:
    python -m pipeline.filter_and_repack \
        --input_dir raw_trajectories/ \
        --output_dir filtered_trajectories/ \
        --min_return 20.0 \
        --gamma 0.99

The input files must be sorted by filename to ensure temporal ordering.
Each file contains arrays with shape (NUM_STEPS, NUM_ENVS, ...).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants matching the Craftax symbolic observation layout
# ---------------------------------------------------------------------------
OBS_DIM = 8268
MAP_DIM = 8217          # first 8217 dims are binary (map section)
AUX_DIM = OBS_DIM - MAP_DIM  # 51 remaining dims (mixed binary + float)
MAP_PACKED_DIM = (MAP_DIM + 7) // 8  # = 1028 bytes after packbits


def bitpack_obs(obs_f16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split float16 obs (N, 8268) into bitpacked map + float16 aux.

    Returns:
        obs_map_bits: uint8, (N, 1028) — packbits of first 8217 binary dims
        obs_aux:      float16, (N, 51) — remaining dims kept as-is
    """
    # Round the map section to 0/1 (they should already be, but guard float16 noise)
    map_section = np.round(obs_f16[:, :MAP_DIM]).astype(np.uint8)
    obs_map_bits = np.packbits(map_section, axis=1, bitorder="little")
    obs_aux = obs_f16[:, MAP_DIM:]
    return obs_map_bits, obs_aux


def compute_episode_rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute return-to-go for a single episode (1-D reward array)."""
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


class EpisodeBuffer:
    """Pre-allocated buffer that accumulates steps for one env."""

    def __init__(self, max_len: int = 30000):
        self.max_len = max_len
        self.obs = np.empty((max_len, OBS_DIM), dtype=np.float16)
        self.action = np.empty(max_len, dtype=np.int32)
        self.reward = np.empty(max_len, dtype=np.float32)
        self.done = np.empty(max_len, dtype=np.bool_)
        self.log_prob = np.empty(max_len, dtype=np.float32)
        self.length = 0

    def append_steps(self, obs, action, reward, done, log_prob):
        """Append a chunk of steps. All arrays have shape (T, ...)."""
        n = obs.shape[0]
        end = self.length + n
        if end > self.max_len:
            # Grow buffer (rare)
            new_max = max(end * 2, self.max_len * 2)
            for attr in ("obs", "action", "reward", "done", "log_prob"):
                old = getattr(self, attr)
                new = np.empty(
                    (new_max,) + old.shape[1:], dtype=old.dtype
                )
                new[: self.length] = old[: self.length]
                setattr(self, attr, new)
            self.max_len = new_max
        self.obs[self.length : end] = obs
        self.action[self.length : end] = action
        self.reward[self.length : end] = reward
        self.done[self.length : end] = done
        self.log_prob[self.length : end] = log_prob
        self.length = end

    def flush(self) -> dict:
        """Return episode data as compact dict and reset."""
        L = self.length
        data = {
            "obs": self.obs[:L].copy(),
            "action": self.action[:L].copy(),
            "reward": self.reward[:L].copy(),
            "done": self.done[:L].copy(),
            "log_prob": self.log_prob[:L].copy(),
        }
        self.length = 0
        return data

    def reset(self):
        self.length = 0


def process_trajectories(
    input_dir: str,
    output_dir: str,
    min_return: float,
    gamma: float,
    samples_per_file: int,
    num_envs_override: int | None = None,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover and sort input files
    files = sorted(input_path.glob("trajectories_batch_*.npz"))
    if not files:
        print(f"No trajectory files found in {input_path}")
        sys.exit(1)
    print(f"Found {len(files)} trajectory files")

    # Detect shape from first file
    with np.load(files[0], allow_pickle=False) as d:
        obs_shape = d["obs"].shape
    if len(obs_shape) == 3:
        num_steps, num_envs = obs_shape[0], obs_shape[1]
    elif len(obs_shape) == 2:
        # Flat (N, obs_dim) — caller must supply --num_envs (legacy default 128).
        num_envs = num_envs_override if num_envs_override is not None else 128
        num_steps = obs_shape[0] // num_envs
        if obs_shape[0] != num_steps * num_envs:
            raise ValueError(
                f"Flat obs of length {obs_shape[0]} not divisible by "
                f"num_envs={num_envs}; pass --num_envs explicitly."
            )
    else:
        raise ValueError(f"Unexpected obs shape: {obs_shape}")
    print(f"Detected layout: num_steps={num_steps}, num_envs={num_envs}")

    # Per-env episode buffers
    buffers = [EpisodeBuffer() for _ in range(num_envs)]

    # Accumulator for output
    out_obs_bits = []
    out_obs_aux = []
    out_action = []
    out_reward = []
    out_done = []
    out_log_prob = []
    out_rtg = []
    total_out_samples = 0
    out_file_idx = 0

    # Stats
    total_episodes = 0
    kept_episodes = 0
    total_transitions = 0
    kept_transitions = 0
    return_histogram = []  # all episode returns for summary

    def flush_output():
        """Write accumulated samples to an output NPZ file."""
        nonlocal out_obs_bits, out_obs_aux, out_action, out_reward
        nonlocal out_done, out_log_prob, out_rtg, total_out_samples, out_file_idx

        if not out_obs_bits:
            return

        obs_bits_arr = np.concatenate(out_obs_bits, axis=0)
        obs_aux_arr = np.concatenate(out_obs_aux, axis=0)
        action_arr = np.concatenate(out_action, axis=0)
        reward_arr = np.concatenate(out_reward, axis=0)
        done_arr = np.concatenate(out_done, axis=0)
        log_prob_arr = np.concatenate(out_log_prob, axis=0)
        rtg_arr = np.concatenate(out_rtg, axis=0)

        out_file = output_path / f"trajectories_{out_file_idx:06d}.npz"
        np.savez_compressed(
            out_file,
            obs_map_bits=obs_bits_arr,
            obs_map_dim=np.int32(MAP_DIM),
            obs_aux=obs_aux_arr,
            action=action_arr,
            reward=reward_arr,
            done=done_arr.astype(np.uint8),
            log_prob=log_prob_arr,
            return_to_go=rtg_arr,
        )
        n = obs_bits_arr.shape[0]
        print(f"  Wrote {out_file.name}: {n} samples")
        total_out_samples += n
        out_file_idx += 1

        # Reset accumulators
        out_obs_bits = []
        out_obs_aux = []
        out_action = []
        out_reward = []
        out_done = []
        out_log_prob = []
        out_rtg = []

    def emit_episode(ep: dict):
        """Bitpack, compute RTG, and add to output accumulator."""
        nonlocal kept_episodes, kept_transitions

        obs_bits, obs_aux = bitpack_obs(ep["obs"])
        rtg = compute_episode_rtg(ep["reward"], gamma)

        out_obs_bits.append(obs_bits)
        out_obs_aux.append(obs_aux)
        out_action.append(ep["action"])
        out_reward.append(ep["reward"])
        out_done.append(ep["done"])
        out_log_prob.append(ep["log_prob"])
        out_rtg.append(rtg)

        kept_episodes += 1
        kept_transitions += len(rtg)

        # Flush if accumulated enough
        acc_samples = sum(a.shape[0] for a in out_obs_bits)
        if acc_samples >= samples_per_file:
            flush_output()

    # Main processing loop
    for fi, fpath in enumerate(files):
        if fi % 500 == 0:
            print(
                f"Processing file {fi}/{len(files)}: {fpath.name} "
                f"(kept {kept_episodes}/{total_episodes} episodes, "
                f"{kept_transitions}/{total_transitions} transitions)"
            )

        with np.load(fpath, allow_pickle=False) as d:
            obs = d["obs"]       # (num_steps, num_envs, 8268) float16
            action = d["action"]  # (num_steps, num_envs)
            reward = d["reward"]  # (num_steps, num_envs)
            done = d["done"]      # (num_steps, num_envs)
            log_prob = d["log_prob"]  # (num_steps, num_envs)

        # Handle flat (N, ...) format from legacy saves
        if len(obs.shape) == 2:
            T = num_steps
            E = num_envs
            obs = obs.reshape(T, E, -1)
            action = action.reshape(T, E)
            reward = reward.reshape(T, E)
            done = done.reshape(T, E)
            log_prob = log_prob.reshape(T, E)

        for env_i in range(num_envs):
            env_obs = obs[:, env_i]           # (64, 8268)
            env_action = action[:, env_i]      # (64,)
            env_reward = reward[:, env_i].astype(np.float32)
            env_done = done[:, env_i]          # (64,)
            env_log_prob = log_prob[:, env_i].astype(np.float32)

            # Find episode boundaries (done=True indices)
            done_indices = np.where(env_done)[0]

            seg_start = 0
            for di in done_indices:
                seg_end = di + 1  # include the done step

                # Append segment to buffer, then flush the complete episode
                buffers[env_i].append_steps(
                    env_obs[seg_start:seg_end],
                    env_action[seg_start:seg_end].astype(np.int32),
                    env_reward[seg_start:seg_end],
                    env_done[seg_start:seg_end],
                    env_log_prob[seg_start:seg_end],
                )

                ep = buffers[env_i].flush()
                ep_return = ep["reward"].sum()
                total_episodes += 1
                total_transitions += len(ep["reward"])
                return_histogram.append(float(ep_return))

                if ep_return >= min_return:
                    emit_episode(ep)

                seg_start = seg_end

            # Remaining steps after last done (ongoing episode)
            if seg_start < num_steps:
                buffers[env_i].append_steps(
                    env_obs[seg_start:],
                    env_action[seg_start:].astype(np.int32),
                    env_reward[seg_start:],
                    env_done[seg_start:],
                    env_log_prob[seg_start:],
                )

    # Flush any remaining accumulated output (don't save incomplete episodes)
    flush_output()

    # Discard incomplete episodes still in buffers
    discarded_incomplete = sum(b.length for b in buffers)

    # Print summary
    returns = np.array(return_histogram) if return_histogram else np.array([0.0])
    print("\n" + "=" * 60)
    print("FILTER & REPACK SUMMARY")
    print("=" * 60)
    print(f"Input files:        {len(files)}")
    print(f"Total episodes:     {total_episodes}")
    print(f"Total transitions:  {total_transitions}")
    print(f"Kept episodes:      {kept_episodes} ({100*kept_episodes/max(total_episodes,1):.1f}%)")
    print(f"Kept transitions:   {kept_transitions} ({100*kept_transitions/max(total_transitions,1):.1f}%)")
    print(f"Discarded incomplete transitions: {discarded_incomplete}")
    print(f"Output files:       {out_file_idx}")
    print(f"Output samples:     {total_out_samples}")
    print(f"\nEpisode return distribution (all {total_episodes} episodes):")
    print(f"  min:    {returns.min():.2f}")
    print(f"  25th:   {np.percentile(returns, 25):.2f}")
    print(f"  median: {np.median(returns):.2f}")
    print(f"  75th:   {np.percentile(returns, 75):.2f}")
    print(f"  max:    {returns.max():.2f}")
    print(f"  mean:   {returns.mean():.2f}")
    print(f"  std:    {returns.std():.2f}")
    print(f"  >= {min_return}: {(returns >= min_return).sum()} ({100*(returns >= min_return).mean():.1f}%)")

    # Histogram buckets
    edges = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100, float("inf")]
    counts, _ = np.histogram(returns, bins=edges)
    print(f"\n  Return histogram:")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"  [{lo:>5.0f}, {hi:>5.0f})" if hi != float("inf") else f"  [{lo:>5.0f},   inf)"
        print(f"  {label}: {counts[i]:>6d}")

    # Save metadata
    meta = {
        "min_return": min_return,
        "gamma": gamma,
        "input_dir": str(input_path),
        "input_files": len(files),
        "num_steps": num_steps,
        "num_envs": num_envs,
        "total_episodes": total_episodes,
        "kept_episodes": kept_episodes,
        "total_transitions": total_transitions,
        "kept_transitions": kept_transitions,
        "discarded_incomplete": discarded_incomplete,
        "output_files": out_file_idx,
        "output_samples": total_out_samples,
        "obs_map_dim": MAP_DIM,
        "obs_aux_dim": AUX_DIM,
        "format": "obs_map_bits(uint8,bitorder=little)+obs_aux(float16)+return_to_go(float32)",
        "return_stats": {
            "min": float(returns.min()),
            "max": float(returns.max()),
            "mean": float(returns.mean()),
            "median": float(np.median(returns)),
            "std": float(returns.std()),
        },
    }
    meta_file = output_path / "filter_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter trajectories by episode return and repack in compressed format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw trajectories_batch_*.npz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write filtered/repacked output",
    )
    parser.add_argument(
        "--min_return",
        type=float,
        default=20.0,
        help="Minimum episode return to keep (default: 20.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for return_to_go (default: 0.99)",
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=100000,
        help="Approximate samples per output NPZ file (default: 100000)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override num_envs for flat (2D) raw NPZs. PPO-RNN uses 1024.",
    )
    args = parser.parse_args()

    process_trajectories(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_return=args.min_return,
        gamma=args.gamma,
        samples_per_file=args.samples_per_file,
        num_envs_override=args.num_envs,
    )


if __name__ == "__main__":
    main()
