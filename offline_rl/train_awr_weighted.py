#!/usr/bin/env python3
"""
Weighted BC+AWR training on imagination-augmented Craftax offline data.

Same as train_awr.py but mixes in oracle (golden) trajectories:
  - Each batch is composed of ORACLE_FRACTION oracle samples + rest from training data
  - Oracle samples receive pure BC loss (maximize log-prob, no advantage weighting)
  - Oracle BC loss is upweighted by ORACLE_LOSS_WEIGHT
  - Regular training samples use standard AWR loss

Usage:
    python -m pipeline.train_awr_weighted \
        --oracle-data /data/group_data/rl/geney/oracle_trajectories.npz \
        --oracle-fraction 0.25 --oracle-loss-weight 5.0
"""

from __future__ import annotations

import os
import glob
import argparse
import json
import numpy as np
import concurrent.futures
import time
from datetime import datetime, timezone

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
    ORACLE_DATA = "/data/group_data/rl/geney/oracle_pipeline/final_trajectories/trajectories_000000.npz"

    ACTION_DIM = 43
    LAYER_WIDTH = 512
    HIDDEN_STATE_DIM = 4096
    OBS_DIM = 8268
    HIDDEN_MODE = "real"
    ADVANTAGE_MODE = "center"
    AUGMENTED = True

    GAMMA = 0.99
    AWR_BETA = 10.0
    AWR_MAX_WEIGHT = 20.0
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOTAL_STEPS = 100_000
    BATCH_SIZE = 256
    LOG_FREQ = 100
    SAVE_FREQ = 25000
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_weighted_bc/"
    SEED = 42
    MAX_DATASET_GB = 80.0

    ORACLE_FRACTION = 0.25
    ORACLE_LOSS_WEIGHT = 5.0
    ENTROPY_COEFF = 0.01
    WEIGHT_DECAY = 0.0
    ANNEAL_ORACLE = False
    VAL_FREQ = 5000

    WANDB_PROJECT = "craftax-offline-awr"
    WANDB_ENTITY = "iris-sobolmark"


# ==============================================================================
# 2. Observation decoding
# ==============================================================================
def decode_obs_array(data) -> np.ndarray:
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

    raise KeyError("No supported observation keys found.")


# ==============================================================================
# 3. Datasets
# ==============================================================================
class OfflineDataset:
    """Standard training dataset (same as train_awr.py)."""
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
        print(f"Training dataset: {len(files)} files, hidden_mode={hidden_mode}")

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

        self._shards = self._build_shards(file_info)
        self._current_shard_idx = 0

        self._norm_count = 0
        self._norm_sum = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float64)
        self._norm_sumsq = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float64)
        self.hidden_mean = np.zeros((Config.HIDDEN_STATE_DIM,), dtype=np.float32)
        self.hidden_std = np.ones((Config.HIDDEN_STATE_DIM,), dtype=np.float32)

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
        print(f"Loading training shard {shard_idx + 1}/{self.num_shards}: "
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

        if self.hidden_state is not None:
            hs64 = self.hidden_state.astype(np.float64)
            self._norm_count += idx
            self._norm_sum += hs64.sum(axis=0)
            self._norm_sumsq += np.square(hs64).sum(axis=0)
            mean = self._norm_sum / max(1, self._norm_count)
            var = np.maximum(self._norm_sumsq / max(1, self._norm_count) - np.square(mean), 1e-6)
            self.hidden_mean = mean.astype(np.float32)
            self.hidden_std = np.sqrt(var).astype(np.float32)

        self._current_shard_idx = shard_idx
        print(f"  Loaded {self.size:,} training samples")

    def save_hidden_stats(self, path: str):
        np.savez(path, mean=self.hidden_mean, std=self.hidden_std)

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
        else:
            hidden_raw = np.zeros((batch_size, Config.HIDDEN_STATE_DIM), dtype=np.float32)

        if self.hidden_mode != "zero":
            hidden_raw = (hidden_raw - self.hidden_mean) / self.hidden_std

        hidden_t = torch.tensor(hidden_raw, dtype=torch.float32, device=Config.DEVICE)

        return {"obs": obs_t, "action": action_t, "hidden_state": hidden_t, "return_to_go": rtg_t}


class OracleDataset:
    """Oracle (golden) trajectory dataset, loaded fully into memory."""
    def __init__(self, oracle_path: str):
        print(f"Loading oracle dataset: {oracle_path}")
        with np.load(oracle_path, allow_pickle=True) as data:
            self.obs = decode_obs_array(data)
            self.action = np.asarray(data["action"]).reshape(-1).astype(np.int32)
            self.return_to_go = np.asarray(data["return_to_go"]).reshape(-1).astype(np.float32)
            raw_h = np.asarray(data["hidden_state"])
            self.hidden_state = (
                np.mean(raw_h, axis=1) if raw_h.ndim == 3 else raw_h
            ).astype(np.float32)

        self.size = len(self.obs)
        print(f"  Oracle samples: {self.size:,}")
        print(f"  Mean RTG: {self.return_to_go.mean():.2f}")

    def sample(self, batch_size: int, hidden_mean: np.ndarray, hidden_std: np.ndarray) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs_t = torch.tensor(self.obs[idx], dtype=torch.float32, device=Config.DEVICE)
        action_t = torch.tensor(self.action[idx], dtype=torch.long, device=Config.DEVICE)
        rtg_t = torch.tensor(self.return_to_go[idx], dtype=torch.float32, device=Config.DEVICE)

        # Oracle hidden states are zero → normalize them the same way
        hidden_raw = (self.hidden_state[idx] - hidden_mean) / hidden_std
        hidden_t = torch.tensor(hidden_raw, dtype=torch.float32, device=Config.DEVICE)

        return {"obs": obs_t, "action": action_t, "hidden_state": hidden_t, "return_to_go": rtg_t}


# ==============================================================================
# 5. Training step (weighted BC+AWR)
# ==============================================================================
def train_step_weighted(
    model, optimizer, train_batch: dict, oracle_batch: dict,
    advantage_mode: str, oracle_loss_weight: float,
    entropy_coeff: float = 0.0,
) -> dict:
    """
    Combined BC+AWR training step.

    - oracle_batch: pure BC loss (maximize log-prob), upweighted
    - train_batch: standard AWR loss (advantage-weighted log-prob)
    - entropy_coeff: entropy bonus to prevent policy collapse
    """
    # --- AWR loss on training data ---
    pi_train, value_train = model(train_batch["obs"], train_batch["hidden_state"])
    advantage = train_batch["return_to_go"] - value_train
    critic_loss = 0.5 * torch.mean(advantage.pow(2))

    log_probs_train = pi_train.log_prob(train_batch["action"])
    adv_detached = advantage.detach()

    if advantage_mode == "center":
        weighted_adv = adv_detached - adv_detached.mean()
    elif advantage_mode == "standardize":
        weighted_adv = (adv_detached - adv_detached.mean()) / adv_detached.std(unbiased=False).clamp_min(1e-6)
    else:
        weighted_adv = adv_detached

    weights = torch.clamp(torch.exp(weighted_adv / Config.AWR_BETA), max=Config.AWR_MAX_WEIGHT)
    awr_loss = -torch.mean(log_probs_train * weights)

    # --- BC loss on oracle data ---
    pi_oracle, value_oracle = model(oracle_batch["obs"], oracle_batch["hidden_state"])
    log_probs_oracle = pi_oracle.log_prob(oracle_batch["action"])
    bc_loss = -torch.mean(log_probs_oracle)

    # Oracle critic loss (also train value head on oracle data)
    oracle_advantage = oracle_batch["return_to_go"] - value_oracle
    oracle_critic_loss = 0.5 * torch.mean(oracle_advantage.pow(2))

    # --- Entropy bonus (across both distributions) ---
    entropy_train = pi_train.entropy().mean()
    entropy_oracle = pi_oracle.entropy().mean()
    entropy_bonus = entropy_train + entropy_oracle

    # --- Combined loss ---
    total_actor_loss = awr_loss + oracle_loss_weight * bc_loss
    total_critic_loss = critic_loss + oracle_critic_loss
    total_loss = total_actor_loss + total_critic_loss - entropy_coeff * entropy_bonus

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        clip_frac = (weights >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage)
        var_ret = torch.var(train_batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_ret + 1e-8))

    return {
        "awr_loss": awr_loss.item(),
        "bc_loss": bc_loss.item(),
        "critic_loss": critic_loss.item(),
        "oracle_critic_loss": oracle_critic_loss.item(),
        "total_loss": total_loss.item(),
        "entropy_train": entropy_train.item(),
        "entropy_oracle": entropy_oracle.item(),
        "entropy_bonus": entropy_bonus.item(),
        "effective_bc_weight": oracle_loss_weight,
        "mean_weight": weights.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value_train": value_train.detach().mean().item(),
        "mean_value_oracle": value_oracle.detach().mean().item(),
        "mean_return_train": train_batch["return_to_go"].mean().item(),
        "mean_return_oracle": oracle_batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "adv_mean": adv_detached.mean().item(),
        "adv_std": adv_detached.std(unbiased=False).item(),
        "hidden_mean": train_batch["hidden_state"].mean().item(),
        "hidden_std": train_batch["hidden_state"].std().item(),
    }


# ==============================================================================
# 5b. Validation (real / zero / shuffled accuracy on held-out data)
# ==============================================================================
@torch.no_grad()
def validate(model, val_data: dict, hidden_mean: np.ndarray, hidden_std: np.ndarray) -> dict:
    """
    Evaluate action prediction accuracy on held-out data in three modes:
      - real: correct hidden states
      - zero: zero hidden states
      - shuffled: randomly permuted hidden states
    """
    model.eval()
    obs_t = torch.tensor(val_data["obs"], dtype=torch.float32, device=Config.DEVICE)
    action_t = torch.tensor(val_data["action"], dtype=torch.long, device=Config.DEVICE)
    n = obs_t.shape[0]

    results = {}
    for mode in ("real", "zero", "shuffled"):
        if mode == "real":
            h = (val_data["hidden_state"] - hidden_mean) / hidden_std
        elif mode == "zero":
            h = np.zeros_like(val_data["hidden_state"])
        else:  # shuffled
            perm = np.random.permutation(n)
            h = (val_data["hidden_state"][perm] - hidden_mean) / hidden_std

        hidden_t = torch.tensor(h, dtype=torch.float32, device=Config.DEVICE)

        # Process in chunks to avoid OOM
        chunk = 4096
        correct = 0
        total_nll = 0.0
        for i in range(0, n, chunk):
            pi, _ = model(obs_t[i:i+chunk], hidden_t[i:i+chunk])
            preds = pi.logits.argmax(dim=-1)
            correct += (preds == action_t[i:i+chunk]).sum().item()
            total_nll += -pi.log_prob(action_t[i:i+chunk]).sum().item()

        results[f"val_acc_{mode}"] = correct / n
        results[f"val_nll_{mode}"] = total_nll / n

    results["val_acc_real_minus_zero"] = results["val_acc_real"] - results["val_acc_zero"]
    model.train()
    return results


def load_val_data(val_path: str) -> dict:
    """Load a validation trajectory file."""
    print(f"Loading validation data: {val_path}")
    with np.load(val_path, allow_pickle=True) as data:
        obs = decode_obs_array(data)
        action = np.asarray(data["action"]).reshape(-1).astype(np.int32)
        raw_h = np.asarray(data["hidden_state"])
        hidden = (np.mean(raw_h, axis=1) if raw_h.ndim == 3 else raw_h).astype(np.float32)
    print(f"  Validation samples: {len(obs):,}")
    return {"obs": obs, "action": action, "hidden_state": hidden}


# ==============================================================================
# 6. CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Weighted BC+AWR training")
    p.add_argument("--data-dir", type=str, default=Config.DATA_DIR)
    p.add_argument("--data-glob", type=str, default=Config.DATA_GLOB)
    p.add_argument("--oracle-data", type=str, default=Config.ORACLE_DATA,
                    help="Path to oracle_trajectories.npz")
    p.add_argument("--max-files", type=int, default=126)
    p.add_argument("--file-offset", type=int, default=0)
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
    p.add_argument("--layer-width", type=int, default=Config.LAYER_WIDTH)
    p.add_argument("--oracle-fraction", type=float, default=Config.ORACLE_FRACTION,
                    help="Fraction of each batch from oracle data (default: 0.25)")
    p.add_argument("--oracle-loss-weight", type=float, default=Config.ORACLE_LOSS_WEIGHT,
                    help="Multiplier for oracle BC loss (default: 5.0)")
    p.add_argument("--entropy-coeff", type=float, default=Config.ENTROPY_COEFF,
                    help="Entropy bonus coefficient (default: 0.01)")
    p.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY,
                    help="AdamW weight decay (default: 0.0)")
    p.add_argument("--anneal-oracle", action="store_true", default=False,
                    help="Linearly anneal oracle loss weight to 0 over training")
    p.add_argument("--val-data", type=str, default=None,
                    help="Path to held-out validation .npz for real/zero/shuffled accuracy")
    p.add_argument("--val-freq", type=int, default=Config.VAL_FREQ,
                    help="Validation frequency in steps (default: 5000)")
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
    Config.ORACLE_FRACTION = args.oracle_fraction
    Config.ORACLE_LOSS_WEIGHT = args.oracle_loss_weight
    Config.ENTROPY_COEFF = args.entropy_coeff
    Config.WEIGHT_DECAY = args.weight_decay
    Config.ANNEAL_ORACLE = args.anneal_oracle
    Config.VAL_FREQ = args.val_freq

    if not Config.AUGMENTED:
        Config.HIDDEN_MODE = "zero"
        args.hidden_mode = "zero"

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    aug_tag = "aug" if Config.AUGMENTED else "unaug"
    anneal_tag = "anneal" if Config.ANNEAL_ORACLE else "const"
    wandb_name = (args.wandb_name or
                  f"bcawr-{aug_tag}-w{Config.LAYER_WIDTH}-"
                  f"of{Config.ORACLE_FRACTION:.2f}-ow{Config.ORACLE_LOSS_WEIGHT:.1f}-"
                  f"ent{Config.ENTROPY_COEFF:.3f}-{anneal_tag}-"
                  f"{time.strftime('%Y%m%d-%H%M%S')}")
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
    print(f"Weighted BC+AWR Training — {'Augmented' if Config.AUGMENTED else 'Unaugmented'} "
          f"(width={Config.LAYER_WIDTH})")
    print("=" * 70)
    print(f"  Device: {Config.DEVICE}")
    print(f"  Oracle fraction: {Config.ORACLE_FRACTION:.0%}")
    print(f"  Oracle loss weight: {Config.ORACLE_LOSS_WEIGHT}x")
    print(f"  Entropy coeff: {Config.ENTROPY_COEFF}")
    print(f"  Weight decay: {Config.WEIGHT_DECAY}")
    print(f"  Anneal oracle: {Config.ANNEAL_ORACLE}")
    print(f"  Data: {args.data_dir} (offset={args.file_offset}, max={args.max_files})")
    print(f"  Oracle: {args.oracle_data}")
    print(f"  Hidden mode: {args.hidden_mode}")
    print(f"  Steps: {Config.TOTAL_STEPS}, Batch: {Config.BATCH_SIZE}, LR: {Config.LR}")
    print(f"  AWR beta: {Config.AWR_BETA}, Advantage: {Config.ADVANTAGE_MODE}")
    if args.val_data:
        print(f"  Validation: {args.val_data} (every {Config.VAL_FREQ} steps)")
    print()

    # Load training dataset
    dataset = OfflineDataset(
        args.data_dir, args.data_glob,
        max_files=args.max_files, file_offset=args.file_offset,
        hidden_mode=args.hidden_mode,
        max_dataset_gb=args.max_dataset_gb,
    )

    # Save normalization stats
    stats_path = os.path.join(Config.SAVE_DIR, "hidden_state_stats.npz")
    dataset.save_hidden_stats(stats_path)
    print(f"Saved hidden normalization stats to {stats_path}")

    # Load oracle dataset
    oracle = OracleDataset(args.oracle_data)

    # Compute batch splits
    oracle_batch_size = max(1, int(Config.BATCH_SIZE * Config.ORACLE_FRACTION))
    train_batch_size = Config.BATCH_SIZE - oracle_batch_size
    print(f"\nBatch composition: {train_batch_size} train + {oracle_batch_size} oracle "
          f"= {Config.BATCH_SIZE} total")

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
    if Config.WEIGHT_DECAY > 0:
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Load validation data
    val_data = None
    if args.val_data:
        val_data = load_val_data(args.val_data)

    # Save metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "augmented": Config.AUGMENTED,
        "layer_width": Config.LAYER_WIDTH,
        "total_params": total_params,
        "dataset_samples": dataset.size,
        "oracle_samples": oracle.size,
        "oracle_fraction": Config.ORACLE_FRACTION,
        "oracle_loss_weight": Config.ORACLE_LOSS_WEIGHT,
        "entropy_coeff": Config.ENTROPY_COEFF,
        "anneal_oracle": Config.ANNEAL_ORACLE,
        "weight_decay": Config.WEIGHT_DECAY,
        "num_shards": dataset.num_shards,
    }
    with open(os.path.join(Config.SAVE_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    model.train()
    t0 = time.time()
    base_oracle_weight = Config.ORACLE_LOSS_WEIGHT

    for step in range(1, Config.TOTAL_STEPS + 1):
        # Shard rotation
        if shard_steps and step > 1 and (step - 1) % shard_steps == 0:
            dataset.advance_shard()
            dataset.save_hidden_stats(stats_path)

        # Compute effective oracle weight (linear anneal if enabled)
        if Config.ANNEAL_ORACLE:
            progress = (step - 1) / max(1, Config.TOTAL_STEPS - 1)
            effective_oracle_weight = base_oracle_weight * (1.0 - progress)
        else:
            effective_oracle_weight = base_oracle_weight

        train_batch = dataset.sample(train_batch_size)
        oracle_batch = oracle.sample(
            oracle_batch_size,
            hidden_mean=dataset.hidden_mean,
            hidden_std=dataset.hidden_std,
        )

        metrics = train_step_weighted(
            model, optimizer, train_batch, oracle_batch,
            Config.ADVANTAGE_MODE, effective_oracle_weight,
            entropy_coeff=Config.ENTROPY_COEFF,
        )

        if step % Config.LOG_FREQ == 0:
            if not args.no_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                elapsed = time.time() - t0
                sps = step / elapsed
                eta_min = (Config.TOTAL_STEPS - step) / sps / 60
                print(
                    f"[{step}/{Config.TOTAL_STEPS}] "
                    f"awr={metrics['awr_loss']:.4f} bc={metrics['bc_loss']:.4f} "
                    f"bc_w={effective_oracle_weight:.2f} "
                    f"critic={metrics['critic_loss']:.4f} "
                    f"expl_var={metrics['explained_variance']:.3f} "
                    f"ent_train={metrics['entropy_train']:.3f} "
                    f"ent_oracle={metrics['entropy_oracle']:.3f} "
                    f"({sps:.0f} steps/s, ETA {eta_min:.0f}min)"
                )

        # Validation
        if val_data is not None and step % Config.VAL_FREQ == 0:
            val_metrics = validate(
                model, val_data,
                hidden_mean=dataset.hidden_mean,
                hidden_std=dataset.hidden_std,
            )
            if not args.no_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            print(
                f"  [Val@{step}] "
                f"acc_real={val_metrics['val_acc_real']:.4f} "
                f"acc_zero={val_metrics['val_acc_zero']:.4f} "
                f"acc_shuf={val_metrics['val_acc_shuffled']:.4f} "
                f"Δ(real-zero)={val_metrics['val_acc_real_minus_zero']:+.4f}"
            )

        if step % Config.SAVE_FREQ == 0:
            ckpt = os.path.join(Config.SAVE_DIR, f"checkpoint_{step}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    final_path = os.path.join(Config.SAVE_DIR, "final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")

    # Final validation
    if val_data is not None:
        val_metrics = validate(
            model, val_data,
            hidden_mean=dataset.hidden_mean,
            hidden_std=dataset.hidden_std,
        )
        print(f"Final validation: "
              f"acc_real={val_metrics['val_acc_real']:.4f} "
              f"acc_zero={val_metrics['val_acc_zero']:.4f} "
              f"acc_shuf={val_metrics['val_acc_shuffled']:.4f} "
              f"Δ(real-zero)={val_metrics['val_acc_real_minus_zero']:+.4f}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
