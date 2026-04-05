#!/usr/bin/env python3
"""
Strict AWR training on imagination-augmented Craftax offline data.

Trains on the first 126 files (80%) from final_trajectories/.
Hidden states are 4096-dim mean-pooled Qwen3-8B layer-30 embeddings,
pre-aligned to each sample by the merge step.

Usage:
    python -m pipeline.train_awr [--total-steps 100000] [--batch-size 256]
"""

from __future__ import annotations

import os
import glob
import argparse
import hashlib
import json
import numpy as np
import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Tuple

try:
    import wandb
except ImportError:
    wandb = None

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.optim as optim

from models.actor_critic_aug import ActorCriticAug, ActorCritic


# ==============================================================================
# 1. Configuration
# ==============================================================================
class Config:
    DATA_DIR = "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
    DATA_GLOB = "trajectories_*.npz"

    ACTION_DIM = 43
    LAYER_WIDTH = 512
    HIDDEN_STATE_DIM = 4096  # Qwen3-8B layer-30 mean-pooled
    OBS_DIM = 8268
    HIDDEN_MODE = "real"  # real | zero | shuffle
    ADVANTAGE_MODE = "center"  # raw | center | standardize
    AUGMENTED = True  # False → obs-only ActorCritic (no hidden branch)

    GAMMA = 0.99
    AWR_BETA = 10.0
    AWR_MAX_WEIGHT = 20.0
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOTAL_STEPS = 100_000
    BATCH_SIZE = 256
    LOG_FREQ = 100
    SAVE_FREQ = 25000
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_imagination/"
    SEED = 42
    MAX_DATASET_GB = 80.0

    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


# ==============================================================================
# 2. Observation decoding
# ==============================================================================
def decode_obs_array(data) -> np.ndarray:
    """Decode observations from NPZ (handles bitpacked and raw formats)."""
    if "obs" in data.files:
        raw_obs = np.asarray(data["obs"])
        if len(raw_obs.shape) > 2:
            raw_obs = raw_obs.reshape(raw_obs.shape[0], -1)
        return raw_obs.astype(np.float32, copy=False)

    if "obs_map_bits" in data.files and "obs_aux" in data.files:
        map_dim = int(data["obs_map_dim"]) if "obs_map_dim" in data.files else 8217
        map_bits = np.asarray(data["obs_map_bits"])
        obs_map = np.unpackbits(
            map_bits, axis=1, count=map_dim, bitorder="little"
        ).astype(np.float32, copy=False)
        obs_aux = np.asarray(data["obs_aux"], dtype=np.float32)
        return np.concatenate([obs_map, obs_aux], axis=1)

    raise KeyError("No supported observation keys found (obs / obs_map_bits+obs_aux).")


# ==============================================================================
# 3. Dataset
# ==============================================================================
class OfflineDataset:
    def __init__(
        self,
        data_dir: str,
        file_pattern: str,
        max_files: int | None = None,
        file_offset: int = 0,
        max_workers: int = 8,
        hidden_mode: str = "real",
        max_dataset_gb: float = 80.0,
    ):
        self.hidden_mode = hidden_mode
        self.max_workers = max_workers
        self.max_dataset_gb = max_dataset_gb
        self.need_hidden = hidden_mode in {"real", "shuffle"}

        search_path = os.path.join(data_dir, file_pattern)
        files = sorted(glob.glob(search_path))
        if not files:
            raise ValueError(f"No files found at {search_path}")
        if file_offset:
            files = files[file_offset:]
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise ValueError(f"No files left after offset={file_offset}, max={max_files}")
        print(f"Dataset: {len(files)} files, hidden_mode={hidden_mode}")

        # Scan for sample counts
        file_info = []
        total_samples = 0
        for f in files:
            try:
                with np.load(f, mmap_mode="r") as d:
                    n = d["reward"].shape[0]
                    total_samples += n
                    file_info.append((f, n))
            except Exception as e:
                print(f"  Skipping corrupt file {f}: {e}")
        if not file_info:
            raise ValueError("No valid files after scanning.")

        self._all_file_info = list(file_info)
        self._bytes_per_sample = (
            4 * Config.OBS_DIM + 4 * Config.HIDDEN_STATE_DIM + 4 + 4
        )
        estimated_gb = self._estimate_gb(total_samples)
        print(f"  Total samples: {total_samples:,}")
        print(f"  Estimated memory: {estimated_gb:.2f} GiB")

        # Build shards
        self._shards = self._build_shards(file_info)
        self._current_shard_idx = 0

        # Normalization stats (accumulated across shards)
        self._norm_count = 0
        self._norm_sum = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float64)
        self._norm_sumsq = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float64)
        self.hidden_mean = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float32)
        self.hidden_std = np.ones((Config.HIDDEN_STATE_DIM,), dtype=np.float32)

        # Buffers (filled by _load_shard)
        self.obs = None
        self.action = None
        self.hidden_state = None
        self.return_to_go = None
        self.size = 0

        if len(self._shards) > 1:
            print(f"  Sharded into {len(self._shards)} chunks for memory budget")
        self._load_shard(0)

    def _estimate_gb(self, n: int) -> float:
        return (n * self._bytes_per_sample) / (1024 ** 3)

    def _build_shards(self, file_info):
        if self.max_dataset_gb is None:
            return [list(file_info)]
        shards, current, current_n = [], [], 0
        for info in file_info:
            n = info[1]
            if current and self._estimate_gb(current_n + n) > self.max_dataset_gb:
                shards.append(current)
                current, current_n = [info], n
            else:
                current.append(info)
                current_n += n
        if current:
            shards.append(current)
        return shards

    @property
    def num_shards(self):
        return len(self._shards)

    def _load_single_file(self, args):
        fpath, _ = args
        try:
            with np.load(fpath, allow_pickle=True) as data:
                obs = decode_obs_array(data)
                action = np.asarray(data["action"]).reshape(-1).astype(np.int32)
                rtg = np.asarray(data["return_to_go"]).reshape(-1).astype(np.float32)

                hidden = None
                if self.need_hidden:
                    raw = np.asarray(data["hidden_state"])
                    hidden = (
                        np.mean(raw, axis=1) if raw.ndim == 3 else raw
                    ).astype(np.float32)

                return {
                    "obs": obs, "action": action, "return_to_go": rtg,
                    "hidden_state": hidden, "count": len(obs),
                }
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")
            return None

    def _load_shard(self, shard_idx: int):
        shard = self._shards[shard_idx]
        n_samples = sum(n for _, n in shard)
        print(f"Loading shard {shard_idx + 1}/{self.num_shards}: "
              f"{len(shard)} files, {n_samples:,} samples")

        self.obs = np.zeros((n_samples, Config.OBS_DIM), dtype=np.float32)
        self.action = np.zeros(n_samples, dtype=np.int32)
        self.return_to_go = np.zeros(n_samples, dtype=np.float32)
        self.hidden_state = (
            np.zeros((n_samples, Config.HIDDEN_STATE_DIM), dtype=np.float32)
            if self.need_hidden else None
        )

        idx = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._load_single_file, info): info for info in shard}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                n = result["count"]
                self.obs[idx:idx + n] = result["obs"]
                self.action[idx:idx + n] = result["action"]
                self.return_to_go[idx:idx + n] = result["return_to_go"]
                if self.hidden_state is not None and result["hidden_state"] is not None:
                    self.hidden_state[idx:idx + n] = result["hidden_state"]
                idx += n

        self.size = idx
        self.obs = self.obs[:idx]
        self.action = self.action[:idx]
        self.return_to_go = self.return_to_go[:idx]
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state[:idx]

        # Update normalization stats
        if self.hidden_state is not None:
            hs64 = self.hidden_state.astype(np.float64)
            self._norm_count += idx
            self._norm_sum += hs64.sum(axis=0)
            self._norm_sumsq += np.square(hs64).sum(axis=0)
            mean = self._norm_sum / max(1, self._norm_count)
            var = np.maximum(self._norm_sumsq / max(1, self._norm_count) - np.square(mean), 1e-6)
            self.hidden_mean = mean.astype(np.float32)
            self.hidden_std = np.sqrt(var).astype(np.float32)
            print(f"  Hidden stats — mean: [{self.hidden_mean.min():.3f}, {self.hidden_mean.max():.3f}], "
                  f"std: [{self.hidden_std.min():.3f}, {self.hidden_std.max():.3f}]")

        self._current_shard_idx = shard_idx
        print(f"  Loaded {self.size:,} samples")

    def save_hidden_stats(self, path: str):
        np.savez(path, mean=self.hidden_mean, std=self.hidden_std)
        print(f"Saved hidden normalization stats to {path}")

    def advance_shard(self) -> bool:
        if self.num_shards <= 1:
            return False
        self._load_shard((self._current_shard_idx + 1) % self.num_shards)
        return True

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)

        obs_t = torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
        action_t = torch.tensor(self.action[idx], dtype=torch.long, device=Config.DEVICE)
        rtg_t = torch.tensor(self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE)

        if self.hidden_mode == "real":
            hidden_raw = self.hidden_state[idx]
        elif self.hidden_mode == "shuffle":
            hidden_raw = self.hidden_state[np.random.permutation(idx)]
        else:  # zero
            hidden_raw = np.zeros((batch_size, Config.HIDDEN_STATE_DIM), dtype=np.float32)

        if self.hidden_mode != "zero":
            hidden_raw = (hidden_raw - self.hidden_mean) / self.hidden_std

        hidden_t = torch.tensor(hidden_raw, dtype=torch.float32, device=Config.DEVICE)

        return {"obs": obs_t, "action": action_t, "hidden_state": hidden_t, "return_to_go": rtg_t}


# ==============================================================================
# 5. Training step
# ==============================================================================
def train_step(model, optimizer, batch, advantage_mode: str) -> dict:
    pi, value = model(batch["obs"], batch["hidden_state"])

    advantage = batch["return_to_go"] - value
    critic_loss = 0.5 * torch.mean(advantage.pow(2))

    log_probs = pi.log_prob(batch["action"])
    adv_detached = advantage.detach()

    if advantage_mode == "center":
        weighted_adv = adv_detached - adv_detached.mean()
    elif advantage_mode == "standardize":
        weighted_adv = (adv_detached - adv_detached.mean()) / adv_detached.std(unbiased=False).clamp_min(1e-6)
    else:  # raw
        weighted_adv = adv_detached

    weights = torch.clamp(torch.exp(weighted_adv / Config.AWR_BETA), max=Config.AWR_MAX_WEIGHT)
    actor_loss = -torch.mean(log_probs * weights)

    optimizer.zero_grad()
    (critic_loss + actor_loss).backward()
    optimizer.step()

    with torch.no_grad():
        clip_frac = (weights >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage)
        var_ret = torch.var(batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_ret + 1e-8))

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": pi.entropy().mean().item(),
        "mean_weight": weights.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value": value.detach().mean().item(),
        "mean_return": batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "adv_mean": adv_detached.mean().item(),
        "adv_std": adv_detached.std(unbiased=False).item(),
        "hidden_mean": batch["hidden_state"].mean().item(),
        "hidden_std": batch["hidden_state"].std().item(),
    }


# ==============================================================================
# 6. CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="AWR training on imagination-augmented data")
    p.add_argument("--data-dir", type=str, default=Config.DATA_DIR)
    p.add_argument("--data-glob", type=str, default=Config.DATA_GLOB)
    p.add_argument("--max-files", type=int, default=126,
                    help="Number of files to train on (default: 126 = 80%% of 158)")
    p.add_argument("--file-offset", type=int, default=0,
                    help="Skip first N files (for train/val split)")
    p.add_argument("--total-steps", type=int, default=Config.TOTAL_STEPS)
    p.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--awr-beta", type=float, default=Config.AWR_BETA)
    p.add_argument("--hidden-mode", type=str, default=Config.HIDDEN_MODE,
                    choices=["real", "zero", "shuffle"])
    p.add_argument("--advantage-mode", type=str, default=Config.ADVANTAGE_MODE,
                    choices=["raw", "center", "standardize"])
    p.add_argument("--max-dataset-gb", type=float, default=Config.MAX_DATASET_GB)
    p.add_argument("--save-dir", type=str, default=Config.SAVE_DIR)
    p.add_argument("--save-freq", type=int, default=Config.SAVE_FREQ)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-augmentation", action="store_true",
                    help="Train obs-only ActorCritic (no hidden state branch)")
    p.add_argument("--layer-width", type=int, default=Config.LAYER_WIDTH,
                    help="Width of hidden layers (default: 512)")
    return p.parse_args()


# ==============================================================================
# 7. Main
# ==============================================================================
def main():
    args = parse_args()

    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    Config.AWR_BETA = args.awr_beta
    Config.TOTAL_STEPS = args.total_steps
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.HIDDEN_MODE = args.hidden_mode
    Config.ADVANTAGE_MODE = args.advantage_mode
    Config.SAVE_DIR = args.save_dir
    Config.SAVE_FREQ = args.save_freq
    Config.SEED = args.seed
    Config.AUGMENTED = not args.no_augmentation
    Config.LAYER_WIDTH = args.layer_width

    # Force hidden_mode=zero when unaugmented (skip loading hidden states)
    if not Config.AUGMENTED:
        Config.HIDDEN_MODE = "zero"
        args.hidden_mode = "zero"

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    aug_tag = "aug" if Config.AUGMENTED else "unaug"
    wandb_name = args.wandb_name or f"awr-{aug_tag}-w{Config.LAYER_WIDTH}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not args.no_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; pass --no-wandb or install it")
        wandb.init(
            project=Config.WANDB_PROJECT, entity=Config.WANDB_ENTITY,
            name=wandb_name,
            config={k: v for k, v in vars(Config).items() if not k.startswith("_")},
            settings=wandb.Settings(init_timeout=300),
        )

    print("=" * 70)
    print(f"AWR Training — {'Augmented' if Config.AUGMENTED else 'Unaugmented'} (width={Config.LAYER_WIDTH})")
    print("=" * 70)
    print(f"  Device: {Config.DEVICE}")
    print(f"  Augmented: {Config.AUGMENTED}")
    print(f"  Layer width: {Config.LAYER_WIDTH}")
    print(f"  Data: {args.data_dir} (offset={args.file_offset}, max={args.max_files})")
    print(f"  Hidden mode: {args.hidden_mode}")
    print(f"  Hidden dim: {Config.HIDDEN_STATE_DIM}")
    print(f"  Steps: {Config.TOTAL_STEPS}, Batch: {Config.BATCH_SIZE}, LR: {Config.LR}")
    print(f"  AWR beta: {Config.AWR_BETA}, Advantage: {Config.ADVANTAGE_MODE}")
    print()

    dataset = OfflineDataset(
        args.data_dir, args.data_glob,
        max_files=args.max_files, file_offset=args.file_offset,
        hidden_mode=args.hidden_mode,
        max_dataset_gb=args.max_dataset_gb,
    )

    # Save normalization stats for validation
    stats_path = os.path.join(Config.SAVE_DIR, "hidden_state_stats.npz")
    dataset.save_hidden_stats(stats_path)

    # Shard rotation schedule
    shard_steps = None
    if dataset.num_shards > 1:
        shard_steps = max(1, Config.TOTAL_STEPS // dataset.num_shards)
        print(f"Shard rotation every {shard_steps} steps ({dataset.num_shards} shards)")

    if Config.AUGMENTED:
        model = ActorCriticAug(
            obs_dim=Config.OBS_DIM,
            action_dim=Config.ACTION_DIM,
            layer_width=Config.LAYER_WIDTH,
            hidden_state_dim=Config.HIDDEN_STATE_DIM,
        ).to(Config.DEVICE)
    else:
        model = ActorCritic(
            obs_dim=Config.OBS_DIM,
            action_dim=Config.ACTION_DIM,
            layer_width=Config.LAYER_WIDTH,
        ).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Save metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "augmented": Config.AUGMENTED,
        "layer_width": Config.LAYER_WIDTH,
        "total_params": total_params,
        "dataset_samples": dataset.size,
        "num_shards": dataset.num_shards,
    }
    with open(os.path.join(Config.SAVE_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    model.train()
    t0 = time.time()

    for step in range(1, Config.TOTAL_STEPS + 1):
        # Shard rotation
        if shard_steps and step > 1 and (step - 1) % shard_steps == 0:
            dataset.advance_shard()
            dataset.save_hidden_stats(stats_path)

        batch = dataset.sample(Config.BATCH_SIZE)
        metrics = train_step(model, optimizer, batch, Config.ADVANTAGE_MODE)

        if step % Config.LOG_FREQ == 0:
            if not args.no_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                elapsed = time.time() - t0
                sps = step / elapsed
                eta_min = (Config.TOTAL_STEPS - step) / sps / 60
                print(
                    f"[{step}/{Config.TOTAL_STEPS}] "
                    f"actor={metrics['actor_loss']:.4f} critic={metrics['critic_loss']:.4f} "
                    f"expl_var={metrics['explained_variance']:.3f} "
                    f"entropy={metrics['entropy']:.3f} "
                    f"({sps:.0f} steps/s, ETA {eta_min:.0f}min)"
                )

        if step % Config.SAVE_FREQ == 0:
            ckpt = os.path.join(Config.SAVE_DIR, f"checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    final_path = os.path.join(Config.SAVE_DIR, "final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
