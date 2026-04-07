#!/usr/bin/env python3
"""
Weighted BC+AWR training v2 — with anti-overfitting features.

Improvements over v1:
  - --entropy-coeff: entropy bonus to prevent policy collapse
  - --anneal-oracle: cosine-anneal oracle_loss_weight from initial to 0
  - --oracle-awr: use AWR loss on oracle data instead of pure BC
  - --weight-decay: AdamW weight decay

Usage:
    python -m pipeline.train_awr_weighted_v2 \
        --oracle-fraction 0.10 --oracle-loss-weight 2.0 \
        --entropy-coeff 0.01
"""

from __future__ import annotations

import os
import glob
import argparse
import json
import math
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

from models.actor_critic_aug import ActorCriticAugLN, ActorCriticAug as ActorCriticAugBase, ActorCriticAugV2, ActorCriticAugGated


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
    SAVE_DIR = "/data/group_data/rl/geney/checkpoints/awr_weighted_bc_v2/"
    SEED = 42
    MAX_DATASET_GB = 60.0

    ORACLE_FRACTION = 0.10
    ORACLE_LOSS_WEIGHT = 2.0
    ENTROPY_COEFF = 0.0
    ANNEAL_ORACLE = False
    ORACLE_AWR = False
    WEIGHT_DECAY = 0.0

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

        hidden_raw = (self.hidden_state[idx] - hidden_mean) / hidden_std
        hidden_t = torch.tensor(hidden_raw, dtype=torch.float32, device=Config.DEVICE)

        return {"obs": obs_t, "action": action_t, "hidden_state": hidden_t, "return_to_go": rtg_t}


# ==============================================================================
# 5. Diagnostics
# ==============================================================================
def compute_gradient_conflict(model, train_batch, oracle_batch, advantage_mode,
                              oracle_loss_weight, oracle_awr) -> dict:
    """Compute gradient cosine similarity between AWR and BC losses.

    Runs two separate backward passes to get per-loss gradients,
    then measures conflict. Does NOT update model weights.
    """
    model.zero_grad()

    # AWR gradient
    pi_train, value_train = model(train_batch["obs"], train_batch["hidden_state"])
    advantage_train = train_batch["return_to_go"] - value_train
    critic_loss = 0.5 * torch.mean(advantage_train.pow(2))
    log_probs_train = pi_train.log_prob(train_batch["action"])
    weights_train = compute_awr_weights(advantage_train, advantage_mode)
    awr_loss = -torch.mean(log_probs_train * weights_train)
    awr_total = awr_loss + critic_loss
    awr_total.backward()

    awr_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            awr_grads[name] = p.grad.detach().clone()

    model.zero_grad()

    # BC gradient
    pi_oracle, value_oracle = model(oracle_batch["obs"], oracle_batch["hidden_state"])
    log_probs_oracle = pi_oracle.log_prob(oracle_batch["action"])
    if oracle_awr:
        advantage_oracle = oracle_batch["return_to_go"] - value_oracle
        weights_oracle = compute_awr_weights(advantage_oracle, advantage_mode)
        bc_actor_loss = -torch.mean(log_probs_oracle * weights_oracle)
    else:
        bc_actor_loss = -torch.mean(log_probs_oracle)
    oracle_advantage = oracle_batch["return_to_go"] - value_oracle
    bc_critic_loss = 0.5 * torch.mean(oracle_advantage.pow(2))
    bc_total = oracle_loss_weight * (bc_actor_loss + bc_critic_loss)
    bc_total.backward()

    bc_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            bc_grads[name] = p.grad.detach().clone()

    model.zero_grad()

    # Compute conflict metrics
    # Whole model
    awr_flat = torch.cat([awr_grads[n].flatten() for n in sorted(awr_grads)])
    bc_flat = torch.cat([bc_grads[n].flatten() for n in sorted(bc_grads)])
    awr_norm = awr_flat.norm().item()
    bc_norm = bc_flat.norm().item()
    cos = torch.nn.functional.cosine_similarity(awr_flat.unsqueeze(0), bc_flat.unsqueeze(0)).item()

    result = {
        "grad_cos_total": cos,
        "grad_norm_awr": awr_norm,
        "grad_norm_bc": bc_norm,
        "grad_ratio_bc_awr": bc_norm / max(awr_norm, 1e-8),
    }

    # Per-submodule: actor obs branch, actor hidden branch, critic
    for prefix, label in [("actor_obs", "actor_obs"), ("actor_h", "actor_hid"),
                          ("critic_obs", "critic_obs"), ("critic_h", "critic_hid"),
                          ("actor_merge", "actor_merge"), ("actor_out", "actor_out")]:
        a_parts = [awr_grads[n].flatten() for n in sorted(awr_grads) if prefix in n]
        b_parts = [bc_grads[n].flatten() for n in sorted(bc_grads) if prefix in n]
        if a_parts and b_parts:
            a = torch.cat(a_parts)
            b = torch.cat(b_parts)
            result[f"grad_cos_{label}"] = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)).item()
            result[f"grad_norm_awr_{label}"] = a.norm().item()
            result[f"grad_norm_bc_{label}"] = b.norm().item()

    return result


# ==============================================================================
# 5b. Training step (weighted BC+AWR v2)
# ==============================================================================
def compute_awr_weights(advantage: torch.Tensor, advantage_mode: str) -> torch.Tensor:
    """Compute AWR advantage weights (shared logic for train and oracle)."""
    adv = advantage.detach()
    if advantage_mode == "center":
        adv = adv - adv.mean()
    elif advantage_mode == "standardize":
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    return torch.clamp(torch.exp(adv / Config.AWR_BETA), max=Config.AWR_MAX_WEIGHT)


def train_step_v2(
    model, optimizer, train_batch: dict, oracle_batch: dict,
    advantage_mode: str, oracle_loss_weight: float,
    entropy_coeff: float, oracle_awr: bool,
) -> dict:
    # --- AWR loss on training data ---
    pi_train, value_train = model(train_batch["obs"], train_batch["hidden_state"])
    advantage_train = train_batch["return_to_go"] - value_train
    critic_loss = 0.5 * torch.mean(advantage_train.pow(2))

    log_probs_train = pi_train.log_prob(train_batch["action"])
    weights_train = compute_awr_weights(advantage_train, advantage_mode)
    awr_loss = -torch.mean(log_probs_train * weights_train)

    # --- Oracle loss ---
    pi_oracle, value_oracle = model(oracle_batch["obs"], oracle_batch["hidden_state"])
    log_probs_oracle = pi_oracle.log_prob(oracle_batch["action"])

    if oracle_awr:
        # AWR on oracle: use advantage weighting (oracle has high RTG, so gets high weights naturally)
        advantage_oracle = oracle_batch["return_to_go"] - value_oracle
        weights_oracle = compute_awr_weights(advantage_oracle, advantage_mode)
        oracle_actor_loss = -torch.mean(log_probs_oracle * weights_oracle)
    else:
        # Pure BC on oracle
        oracle_actor_loss = -torch.mean(log_probs_oracle)

    oracle_advantage = oracle_batch["return_to_go"] - value_oracle
    oracle_critic_loss = 0.5 * torch.mean(oracle_advantage.pow(2))

    # --- Entropy bonus ---
    entropy_train = pi_train.entropy().mean()
    entropy_oracle = pi_oracle.entropy().mean()
    entropy_loss = -entropy_train  # only regularize training stream

    # --- Combined loss ---
    total_actor_loss = awr_loss + oracle_loss_weight * oracle_actor_loss
    total_critic_loss = critic_loss + oracle_loss_weight * oracle_critic_loss
    total_loss = total_actor_loss + total_critic_loss + entropy_coeff * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        clip_frac = (weights_train >= Config.AWR_MAX_WEIGHT).float().mean().item()
        var_diff = torch.var(advantage_train)
        var_ret = torch.var(train_batch["return_to_go"])
        explained_var = 1.0 - (var_diff / (var_ret + 1e-8))
        adv_train_np = advantage_train.detach().cpu().numpy()
        weights_train_np = weights_train.detach().cpu().numpy()

    result = {
        "actor_loss": awr_loss.item(),
        "oracle_actor_loss": oracle_actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "oracle_critic_loss": oracle_critic_loss.item(),
        "total_loss": total_loss.item(),
        "entropy_train": entropy_train.item(),
        "entropy_oracle": entropy_oracle.item(),
        "entropy_loss": entropy_loss.item(),
        "oracle_loss_weight_effective": oracle_loss_weight,
        "mean_weight_train": weights_train.mean().item(),
        "weight_clip_frac": clip_frac,
        "mean_value_train": value_train.detach().mean().item(),
        "mean_value_oracle": value_oracle.detach().mean().item(),
        "mean_return_train": train_batch["return_to_go"].mean().item(),
        "mean_return_oracle": oracle_batch["return_to_go"].mean().item(),
        "explained_variance": explained_var.item(),
        "adv_mean": advantage_train.detach().mean().item(),
        "adv_std": advantage_train.detach().std(unbiased=False).item(),
        "_adv_histogram": adv_train_np,
        "_weight_histogram": weights_train_np,
    }

    # Log gate values if using gated architecture
    if hasattr(model, "_last_actor_gate"):
        result["gate_actor_mean"] = model._last_actor_gate.mean().item()
        result["gate_actor_std"] = model._last_actor_gate.std().item()
        result["gate_critic_mean"] = model._last_critic_gate.mean().item()

    return result


# ==============================================================================
# 5c. Hidden-source separability check
# ==============================================================================
def check_hidden_separability(train_hidden: np.ndarray, oracle_hidden: np.ndarray,
                               hidden_mean: np.ndarray, hidden_std: np.ndarray) -> dict:
    """Check if hidden vectors statistically separate golden vs training data.

    If they do, the model could use the hidden branch as a dataset-source tag
    rather than a semantic signal.
    """
    # Normalize both with training stats
    t = (train_hidden - hidden_mean) / hidden_std
    o = (oracle_hidden - hidden_mean) / hidden_std

    # Sample to keep it fast
    n = min(2000, len(t), len(o))
    rng = np.random.RandomState(42)
    t_sample = t[rng.choice(len(t), n, replace=False)]
    o_sample = o[rng.choice(len(o), n, replace=False)]

    # Mean distance between sources
    t_mean = t_sample.mean(axis=0)
    o_mean = o_sample.mean(axis=0)
    mean_l2 = float(np.linalg.norm(t_mean - o_mean))
    mean_cos = float(np.dot(t_mean, o_mean) / (np.linalg.norm(t_mean) * np.linalg.norm(o_mean) + 1e-8))

    # Norms
    t_norms = np.linalg.norm(t_sample, axis=1)
    o_norms = np.linalg.norm(o_sample, axis=1)

    # Cross-source vs within-source cosine sim
    within_t = []
    within_o = []
    cross = []
    for _ in range(500):
        i, j = rng.randint(0, n, 2)
        if i != j:
            within_t.append(np.dot(t_sample[i], t_sample[j]) / (t_norms[i] * t_norms[j] + 1e-8))
        i, j = rng.randint(0, n, 2)
        if i != j:
            within_o.append(np.dot(o_sample[i], o_sample[j]) / (o_norms[i] * o_norms[j] + 1e-8))
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        cross.append(np.dot(t_sample[i], o_sample[j]) / (t_norms[i] * o_norms[j] + 1e-8))

    return {
        "source_mean_l2": mean_l2,
        "source_mean_cos": mean_cos,
        "source_within_train_cos": float(np.mean(within_t)),
        "source_within_oracle_cos": float(np.mean(within_o)),
        "source_cross_cos": float(np.mean(cross)),
        "source_train_norm_mean": float(t_norms.mean()),
        "source_oracle_norm_mean": float(o_norms.mean()),
    }


# ==============================================================================
# 5d. Validation (counterfactual embedding evaluation)
# ==============================================================================
@torch.no_grad()
def validate(
    model, val_data: dict, hidden_mean: np.ndarray, hidden_std: np.ndarray,
    train_hidden: np.ndarray | None = None,
) -> dict:
    """
    Evaluate action prediction accuracy on held-out data in multiple modes:
      real, zero, shuffled, mean_emb, random_train
    Also computes: KL(real||mode), action agreement(real, mode), logit L2 shift.
    """
    model.eval()
    obs_t = torch.tensor(val_data["obs"], dtype=torch.float32, device=Config.DEVICE)
    action_t = torch.tensor(val_data["action"], dtype=torch.long, device=Config.DEVICE)
    n = obs_t.shape[0]
    dim = val_data["hidden_state"].shape[1]

    # Build all hidden conditions
    h_real = (val_data["hidden_state"] - hidden_mean) / hidden_std
    h_mean_emb = np.broadcast_to(h_real.mean(axis=0, keepdims=True), (n, dim)).copy()
    conditions = {
        "real": h_real,
        "zero": np.zeros((n, dim), dtype=np.float32),
        "shuffled": h_real[np.random.permutation(n)],
        "mean_emb": h_mean_emb,
    }
    if train_hidden is not None:
        idx = np.random.choice(len(train_hidden), size=n, replace=len(train_hidden) < n)
        conditions["random_train"] = (train_hidden[idx] - hidden_mean) / hidden_std

    # Collect logits per condition
    chunk = 4096
    all_logits = {}
    for mode, h in conditions.items():
        hidden_t = torch.tensor(h, dtype=torch.float32, device=Config.DEVICE)
        logits_list = []
        for i in range(0, n, chunk):
            pi, _ = model(obs_t[i:i+chunk], hidden_t[i:i+chunk])
            logits_list.append(pi.logits)
        all_logits[mode] = torch.cat(logits_list, dim=0)

    # Compute metrics from logits
    results = {}
    real_logits = all_logits["real"]
    real_probs = torch.softmax(real_logits, dim=-1)
    real_log_probs = torch.log_softmax(real_logits, dim=-1)
    real_preds = real_logits.argmax(dim=-1)

    for mode, logits in all_logits.items():
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # Accuracy on oracle actions
        results[f"acc_{mode}"] = (preds == action_t).float().mean().item()
        # NLL on oracle actions
        results[f"nll_{mode}"] = -log_probs[torch.arange(n), action_t].mean().item()

        if mode != "real":
            # Action agreement with real-embedding policy
            results[f"agree_{mode}"] = (preds == real_preds).float().mean().item()
            # KL(real || mode): how different is the policy under this condition?
            kl = (real_probs * (real_log_probs - log_probs)).sum(dim=-1).mean().item()
            results[f"kl_real_vs_{mode}"] = kl
            # Mean L2 logit shift
            logit_shift = (logits - real_logits).pow(2).mean().item()
            results[f"logit_l2_{mode}"] = logit_shift

    results["acc_real_minus_zero"] = results["acc_real"] - results["acc_zero"]
    if "acc_random_train" in results:
        results["acc_real_minus_random"] = results["acc_real"] - results["acc_random_train"]
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
    p = argparse.ArgumentParser(description="Weighted BC+AWR training v2")
    p.add_argument("--data-dir", type=str, default=Config.DATA_DIR)
    p.add_argument("--data-glob", type=str, default=Config.DATA_GLOB)
    p.add_argument("--oracle-data", type=str, default=Config.ORACLE_DATA)
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
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--save-freq", type=int, default=Config.SAVE_FREQ)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--layer-width", type=int, default=Config.LAYER_WIDTH)
    # Oracle mixing
    p.add_argument("--oracle-fraction", type=float, default=Config.ORACLE_FRACTION)
    p.add_argument("--oracle-loss-weight", type=float, default=Config.ORACLE_LOSS_WEIGHT)
    # Anti-overfitting
    p.add_argument("--entropy-coeff", type=float, default=Config.ENTROPY_COEFF,
                    help="Entropy bonus coefficient (default: 0.0)")
    p.add_argument("--anneal-oracle", action="store_true",
                    help="Cosine-anneal oracle_loss_weight from initial to 0")
    p.add_argument("--oracle-awr", action="store_true",
                    help="Use AWR loss on oracle data instead of pure BC")
    p.add_argument("--weight-decay", type=float, default=Config.WEIGHT_DECAY,
                    help="AdamW weight decay (default: 0.0)")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate for all hidden layers (default: 0.0)")
    p.add_argument("--no-layernorm", action="store_true",
                    help="Use ActorCriticAug (no LayerNorm) instead of ActorCriticAugLN")
    p.add_argument("--arch-v2", action="store_true",
                    help="Use ActorCriticAugV2 (deep obs branch + late hidden injection)")
    p.add_argument("--arch-gated", action="store_true",
                    help="Use ActorCriticAugGated (V2 + learned 0/1 gate on imagination)")
    p.add_argument("--val-data", type=str, default=None,
                    help="Path to held-out validation .npz for real/zero/shuffled accuracy")
    p.add_argument("--val-freq", type=int, default=5000,
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
    Config.LAYER_WIDTH = args.layer_width
    Config.ORACLE_FRACTION = args.oracle_fraction
    Config.ORACLE_LOSS_WEIGHT = args.oracle_loss_weight
    Config.ENTROPY_COEFF = args.entropy_coeff
    Config.ANNEAL_ORACLE = args.anneal_oracle
    Config.ORACLE_AWR = args.oracle_awr
    Config.WEIGHT_DECAY = args.weight_decay

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    variant_tag = []
    if args.oracle_awr:
        variant_tag.append("oawr")
    if args.anneal_oracle:
        variant_tag.append("anneal")
    if args.entropy_coeff > 0:
        variant_tag.append(f"ent{args.entropy_coeff}")
    if args.weight_decay > 0:
        variant_tag.append(f"wd{args.weight_decay}")
    if args.dropout > 0:
        variant_tag.append(f"drop{args.dropout}")
    variant_str = "-".join(variant_tag) if variant_tag else "gentle"

    wandb_name = (args.wandb_name or
                  f"bcawr-v2-{variant_str}-of{Config.ORACLE_FRACTION:.2f}-"
                  f"ow{Config.ORACLE_LOSS_WEIGHT:.1f}")
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
    print(f"Weighted BC+AWR v2 Training (width={Config.LAYER_WIDTH})")
    print("=" * 70)
    print(f"  Device: {Config.DEVICE}")
    print(f"  Oracle fraction: {Config.ORACLE_FRACTION:.0%}")
    print(f"  Oracle loss weight: {Config.ORACLE_LOSS_WEIGHT}x"
          f"{' (cosine annealing)' if Config.ANNEAL_ORACLE else ''}")
    print(f"  Oracle loss type: {'AWR' if Config.ORACLE_AWR else 'BC'}")
    print(f"  Entropy coeff: {Config.ENTROPY_COEFF}")
    print(f"  Weight decay: {Config.WEIGHT_DECAY}")
    print(f"  Dropout: {args.dropout}")
    print(f"  LayerNorm: {not args.no_layernorm}")
    print(f"  Data: {args.data_dir} (offset={args.file_offset}, max={args.max_files})")
    print(f"  Oracle: {args.oracle_data}")
    print(f"  Steps: {Config.TOTAL_STEPS}, Batch: {Config.BATCH_SIZE}, LR: {Config.LR}")
    print(f"  AWR beta: {Config.AWR_BETA}, Advantage: {Config.ADVANTAGE_MODE}")
    if args.val_data:
        print(f"  Validation: {args.val_data} (every {args.val_freq} steps)")
    print()

    # Load training dataset
    dataset = OfflineDataset(
        args.data_dir, args.data_glob,
        max_files=args.max_files, file_offset=args.file_offset,
        hidden_mode=args.hidden_mode,
        max_dataset_gb=args.max_dataset_gb,
    )

    stats_path = os.path.join(Config.SAVE_DIR, "hidden_state_stats.npz")
    dataset.save_hidden_stats(stats_path)

    # Load oracle dataset
    oracle = OracleDataset(args.oracle_data)

    # Batch splits
    oracle_batch_size = max(1, int(Config.BATCH_SIZE * Config.ORACLE_FRACTION))
    train_batch_size = Config.BATCH_SIZE - oracle_batch_size
    print(f"\nBatch composition: {train_batch_size} train + {oracle_batch_size} oracle "
          f"= {Config.BATCH_SIZE} total")

    # Shard rotation
    shard_steps = None
    if dataset.num_shards > 1:
        shard_steps = max(1, Config.TOTAL_STEPS // dataset.num_shards)
        print(f"Shard rotation every {shard_steps} steps ({dataset.num_shards} shards)")

    if args.arch_gated:
        ModelClass = ActorCriticAugGated
    elif args.arch_v2:
        ModelClass = ActorCriticAugV2
    elif args.no_layernorm:
        ModelClass = ActorCriticAugBase
    else:
        ModelClass = ActorCriticAugLN
    model_kwargs = dict(
        obs_dim=Config.OBS_DIM,
        action_dim=Config.ACTION_DIM,
        layer_width=Config.LAYER_WIDTH,
        hidden_state_dim=Config.HIDDEN_STATE_DIM,
    )
    if ModelClass in (ActorCriticAugLN, ActorCriticAugV2, ActorCriticAugGated):
        model_kwargs["dropout"] = args.dropout
    model = ModelClass(**model_kwargs).to(Config.DEVICE)

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
        "arch": ModelClass.__name__,
        "layer_width": Config.LAYER_WIDTH,
        "total_params": total_params,
        "dataset_samples": dataset.size,
        "oracle_samples": oracle.size,
        "oracle_fraction": Config.ORACLE_FRACTION,
        "oracle_loss_weight": Config.ORACLE_LOSS_WEIGHT,
        "entropy_coeff": Config.ENTROPY_COEFF,
        "anneal_oracle": Config.ANNEAL_ORACLE,
        "oracle_awr": Config.ORACLE_AWR,
        "weight_decay": Config.WEIGHT_DECAY,
        "dropout": args.dropout,
        "no_layernorm": args.no_layernorm,
        "num_shards": dataset.num_shards,
    }
    with open(os.path.join(Config.SAVE_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # --- Pre-training diagnostics ---
    if dataset.hidden_state is not None:
        sep = check_hidden_separability(
            dataset.hidden_state, oracle.hidden_state,
            dataset.hidden_mean, dataset.hidden_std,
        )
        print("\n--- Hidden-source separability ---")
        print(f"  Mean L2 distance: {sep['source_mean_l2']:.4f}")
        print(f"  Mean cosine: {sep['source_mean_cos']:.4f}")
        print(f"  Within-train cos: {sep['source_within_train_cos']:.4f}")
        print(f"  Within-oracle cos: {sep['source_within_oracle_cos']:.4f}")
        print(f"  Cross-source cos: {sep['source_cross_cos']:.4f}")
        print(f"  Train norm: {sep['source_train_norm_mean']:.2f}  Oracle norm: {sep['source_oracle_norm_mean']:.2f}")
        if not args.no_wandb:
            wandb.log({f"diag/{k}": v for k, v in sep.items()}, step=0)

    model.train()
    t0 = time.time()

    for step in range(1, Config.TOTAL_STEPS + 1):
        # Shard rotation
        if shard_steps and step > 1 and (step - 1) % shard_steps == 0:
            dataset.advance_shard()
            dataset.save_hidden_stats(stats_path)

        # Compute effective oracle weight (with optional annealing)
        if Config.ANNEAL_ORACLE:
            progress = step / Config.TOTAL_STEPS
            effective_ow = Config.ORACLE_LOSS_WEIGHT * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            effective_ow = Config.ORACLE_LOSS_WEIGHT

        train_batch = dataset.sample(train_batch_size)
        oracle_batch = oracle.sample(
            oracle_batch_size,
            hidden_mean=dataset.hidden_mean,
            hidden_std=dataset.hidden_std,
        )

        metrics = train_step_v2(
            model, optimizer, train_batch, oracle_batch,
            Config.ADVANTAGE_MODE, effective_ow,
            Config.ENTROPY_COEFF, Config.ORACLE_AWR,
        )

        if step % Config.LOG_FREQ == 0:
            if not args.no_wandb:
                log_dict = {f"train/{k}": v for k, v in metrics.items()
                            if not k.startswith("_")}
                # Log histograms every 10x log freq
                if step % (Config.LOG_FREQ * 10) == 0:
                    if "_adv_histogram" in metrics:
                        log_dict["train/adv_histogram"] = wandb.Histogram(metrics["_adv_histogram"])
                    if "_weight_histogram" in metrics:
                        log_dict["train/weight_histogram"] = wandb.Histogram(metrics["_weight_histogram"])
                wandb.log(log_dict, step=step)
            if step % (Config.LOG_FREQ * 10) == 0:
                elapsed = time.time() - t0
                sps = step / elapsed
                eta_min = (Config.TOTAL_STEPS - step) / sps / 60
                print(
                    f"[{step}/{Config.TOTAL_STEPS}] "
                    f"awr={metrics['actor_loss']:.4f} "
                    f"oracle={metrics['oracle_actor_loss']:.4f} "
                    f"ow_eff={effective_ow:.2f} "
                    f"critic={metrics['critic_loss']:.4f} "
                    f"expl_var={metrics['explained_variance']:.3f} "
                    f"ent_t={metrics['entropy_train']:.3f} "
                    f"ent_o={metrics['entropy_oracle']:.3f} "
                    f"({sps:.0f} sps, ETA {eta_min:.0f}min)"
                )

        # Validation + diagnostics
        if val_data is not None and step % args.val_freq == 0:
            val_metrics = validate(
                model, val_data,
                hidden_mean=dataset.hidden_mean,
                hidden_std=dataset.hidden_std,
                train_hidden=dataset.hidden_state,
            )
            if not args.no_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            rt_str = ""
            if "acc_random_train" in val_metrics:
                rt_str = f" acc_rand={val_metrics['acc_random_train']:.4f}"
            mean_str = ""
            if "acc_mean_emb" in val_metrics:
                mean_str = f" acc_mean={val_metrics['acc_mean_emb']:.4f}"
            print(
                f"  [Val@{step}] "
                f"acc_real={val_metrics['acc_real']:.4f} "
                f"acc_zero={val_metrics['acc_zero']:.4f} "
                f"acc_shuf={val_metrics['acc_shuffled']:.4f}"
                f"{rt_str}{mean_str} "
                f"Δ(real-zero)={val_metrics['acc_real_minus_zero']:+.4f}"
            )
            # KL and agreement
            for mode in ["zero", "shuffled", "mean_emb"]:
                kl_key = f"kl_real_vs_{mode}"
                ag_key = f"agree_{mode}"
                if kl_key in val_metrics:
                    print(f"           KL(real||{mode})={val_metrics[kl_key]:.4f}  "
                          f"agree={val_metrics[ag_key]:.4f}  "
                          f"logit_l2={val_metrics.get(f'logit_l2_{mode}', 0):.4f}")

            # Gradient conflict diagnostic (every 2x val_freq to save time)
            if effective_ow > 0 and step % (args.val_freq * 2) == 0:
                grad_metrics = compute_gradient_conflict(
                    model, train_batch, oracle_batch,
                    Config.ADVANTAGE_MODE, effective_ow, Config.ORACLE_AWR,
                )
                if not args.no_wandb:
                    wandb.log({f"grad/{k}": v for k, v in grad_metrics.items()}, step=step)
                print(f"           grad_cos={grad_metrics['grad_cos_total']:+.4f}  "
                      f"awr_norm={grad_metrics['grad_norm_awr']:.4f}  "
                      f"bc_norm={grad_metrics['grad_norm_bc']:.4f}  "
                      f"ratio={grad_metrics['grad_ratio_bc_awr']:.2f}")

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
            train_hidden=dataset.hidden_state,
        )
        rt_str = ""
        if "acc_random_train" in val_metrics:
            rt_str = f" acc_rand={val_metrics['acc_random_train']:.4f}"
        print(f"Final validation: "
              f"acc_real={val_metrics['acc_real']:.4f} "
              f"acc_zero={val_metrics['acc_zero']:.4f} "
              f"acc_shuf={val_metrics['acc_shuffled']:.4f}"
              f"{rt_str} "
              f"Δ(real-zero)={val_metrics['acc_real_minus_zero']:+.4f}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
