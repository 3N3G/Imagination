#!/usr/bin/env python3
"""
Validate a trained AWR model on held-out data.

Computes policy negative log-likelihood under three conditions:
  1. real    — actual hidden states (normalized)
  2. zero    — zero hidden states
  3. shuffled — randomly permuted hidden states (sample i gets hidden j)

If the model learned to use the semantic content of the embeddings:
  - real should have the lowest NLL (best)
  - shuffled should be worse than real (wrong context confuses the model)
  - zero should also be worse than real

Usage:
    python -m pipeline.validate_awr --checkpoint /path/to/final.pth
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from models.actor_critic_aug import (
    ActorCriticAugLN,
    ActorCriticAug as ActorCriticAugBase,
    ActorCriticAugV2,
    ActorCritic,
    ActorCriticHiddenOnly,
)


# ==============================================================================
# Config (must match training)
# ==============================================================================
ACTION_DIM = 43
DEFAULT_LAYER_WIDTH = 512
HIDDEN_STATE_DIM = 4096
OBS_DIM = 8268

DEFAULT_DATA_DIR = (
    "/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards/final_trajectories"
)
DEFAULT_STATS_DIR = "/data/group_data/rl/geney/checkpoints/awr_imagination/"


# ==============================================================================
# Observation decoding (same as train_awr.py)
# ==============================================================================
def decode_obs_array(data) -> np.ndarray:
    if "obs" in data.files:
        raw = np.asarray(data["obs"])
        if raw.ndim > 2:
            raw = raw.reshape(raw.shape[0], -1)
        return raw.astype(np.float32, copy=False)
    if "obs_map_bits" in data.files and "obs_aux" in data.files:
        map_dim = int(data["obs_map_dim"]) if "obs_map_dim" in data.files else 8217
        bits = np.asarray(data["obs_map_bits"])
        obs_map = np.unpackbits(bits, axis=1, count=map_dim, bitorder="little").astype(
            np.float32, copy=False
        )
        obs_aux = np.asarray(data["obs_aux"], dtype=np.float32)
        return np.concatenate([obs_map, obs_aux], axis=1)
    raise KeyError("No supported observation keys.")


# ==============================================================================
# Data loading
# ==============================================================================
def load_val_data(
    data_dir: str,
    file_pattern: str,
    file_offset: int,
    max_files: int | None,
) -> dict:
    """Load all validation files into memory."""
    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if file_offset:
        files = files[file_offset:]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No val files found (offset={file_offset}, max={max_files})")

    print(f"Loading {len(files)} validation files...")
    all_obs, all_action, all_hidden, all_rtg = [], [], [], []

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            all_obs.append(decode_obs_array(data))
            all_action.append(np.asarray(data["action"]).reshape(-1).astype(np.int32))
            raw_h = np.asarray(data["hidden_state"])
            all_hidden.append(
                np.mean(raw_h, axis=1).astype(np.float32)
                if raw_h.ndim == 3
                else raw_h.astype(np.float32)
            )
            all_rtg.append(np.asarray(data["return_to_go"]).reshape(-1).astype(np.float32))

    obs = np.concatenate(all_obs)
    action = np.concatenate(all_action)
    hidden = np.concatenate(all_hidden)
    rtg = np.concatenate(all_rtg)
    print(f"  Loaded {len(obs):,} validation samples")
    return {"obs": obs, "action": action, "hidden_state": hidden, "return_to_go": rtg}


# ==============================================================================
# Evaluation
# ==============================================================================
@torch.no_grad()
def evaluate_nll(
    model: ActorCriticAug,
    obs: np.ndarray,
    action: np.ndarray,
    hidden: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> dict:
    """Compute mean NLL and accuracy over the full dataset."""
    n = len(obs)
    total_nll = 0.0
    total_correct = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_t = torch.tensor(obs[start:end], dtype=torch.float32, device=device)
        act_t = torch.tensor(action[start:end], dtype=torch.long, device=device)
        hid_t = torch.tensor(hidden[start:end], dtype=torch.float32, device=device)

        pi, _ = model(obs_t, hid_t)
        log_p = pi.log_prob(act_t)
        total_nll -= log_p.sum().item()
        total_correct += (pi.logits.argmax(dim=-1) == act_t).sum().item()

    mean_nll = total_nll / n
    accuracy = total_correct / n
    return {"mean_nll": mean_nll, "accuracy": accuracy, "n_samples": n}


def build_hidden_conditions(
    hidden_real: np.ndarray,
    hidden_mean: np.ndarray,
    hidden_std: np.ndarray,
    seed: int = 42,
) -> dict:
    """Build the three hidden-state conditions for evaluation."""
    n, dim = hidden_real.shape

    # Real: normalize with training stats
    real_normed = (hidden_real - hidden_mean) / hidden_std

    # Zero: zeros (no hidden information)
    zero = np.zeros((n, dim), dtype=np.float32)

    # Shuffled: random permutation (same distribution, wrong pairing)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    shuffled_normed = (hidden_real[perm] - hidden_mean) / hidden_std

    return {"real": real_normed, "zero": zero, "shuffled": shuffled_normed}


# ==============================================================================
# CLI & Main
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Validate AWR model on held-out data")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained model checkpoint (.pth)")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--data-glob", type=str, default="trajectories_*.npz")
    p.add_argument("--file-offset", type=int, default=126,
                    help="Skip first N files (default: 126, i.e. val = files 126-157)")
    p.add_argument("--max-files", type=int, default=32,
                    help="Number of val files (default: 32)")
    p.add_argument("--hidden-stats", type=str,
                    default=os.path.join(DEFAULT_STATS_DIR, "hidden_state_stats.npz"),
                    help="Path to hidden_state_stats.npz from training")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layer-width", type=int, default=DEFAULT_LAYER_WIDTH,
                    help="Width of hidden layers (must match training)")
    p.add_argument("--no-augmentation", action="store_true",
                    help="Use obs-only ActorCritic (no hidden state branch)")
    p.add_argument("--no-layernorm", action="store_true",
                    help="Use ActorCriticAug (no LayerNorm) instead of ActorCriticAugLN")
    p.add_argument("--arch-v2", action="store_true",
                    help="Use ActorCriticAugV2 architecture")
    p.add_argument("--arch-hidden-only", action="store_true",
                    help="Use ActorCriticHiddenOnly (hidden/imagination input only)")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate (must match training architecture)")
    p.add_argument("--cross-hidden-dir", type=str, default=None,
                    help="Load hidden states from a DIFFERENT data dir (same obs/actions) "
                         "to test cross-distribution embedding transfer")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_STATE_DIM,
                    help="Hidden state dim of the trained model (default 4096 for Qwen3-8B; "
                         "set to 3072 for Gemini embedding-001)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("AWR Validation — Hidden State Ablation")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Val data: {args.data_dir} (offset={args.file_offset}, max={args.max_files})")
    print(f"  Device: {args.device}")
    print()

    augmented = not args.no_augmentation
    layer_width = args.layer_width

    # Load model
    if augmented:
        if args.arch_hidden_only:
            model = ActorCriticHiddenOnly(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                layer_width=layer_width,
                hidden_state_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(args.device)
        elif args.arch_v2:
            model = ActorCriticAugV2(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                layer_width=layer_width,
                hidden_state_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(args.device)
        elif args.no_layernorm:
            model = ActorCriticAugBase(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                layer_width=layer_width,
                hidden_state_dim=args.hidden_dim,
            ).to(args.device)
        else:
            model = ActorCriticAugLN(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                layer_width=layer_width,
                hidden_state_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(args.device)
    else:
        model = ActorCritic(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            layer_width=layer_width,
        ).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model_type = "ActorCriticAug" if augmented else "ActorCritic"
    print(f"  Model loaded: {model_type} (width={layer_width}, "
          f"{sum(p.numel() for p in model.parameters()):,} params)")

    # Load validation data
    val = load_val_data(args.data_dir, args.data_glob, args.file_offset, args.max_files)

    print()
    print("=" * 70)
    print(f"{'Condition':<12} {'Mean NLL':>10} {'Accuracy':>10} {'Samples':>10}")
    print("-" * 70)

    results = {}

    if augmented:
        # Load normalization stats from training
        stats = np.load(args.hidden_stats)
        hidden_mean = stats["mean"].astype(np.float32)
        hidden_std = stats["std"].astype(np.float32)
        print(f"  Hidden stats loaded from {args.hidden_stats}")

        # Build three conditions
        conditions = build_hidden_conditions(val["hidden_state"], hidden_mean, hidden_std, args.seed)

        # Optionally add cross-distribution condition (e.g. PSF embeddings on oracle model)
        if args.cross_hidden_dir:
            print(f"  Loading cross-distribution hidden states from {args.cross_hidden_dir}")
            cross_val = load_val_data(
                args.cross_hidden_dir, args.data_glob,
                args.file_offset, args.max_files,
            )
            assert len(cross_val["hidden_state"]) == len(val["hidden_state"]), \
                "Cross hidden states must have same number of samples"
            cross_normed = (cross_val["hidden_state"] - hidden_mean) / hidden_std
            conditions["cross"] = cross_normed
            del cross_val

        for name, hidden in conditions.items():
            t0 = time.time()
            metrics = evaluate_nll(
                model, val["obs"], val["action"], hidden,
                device=args.device, batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            results[name] = metrics
            print(
                f"{name:<12} {metrics['mean_nll']:>10.4f} {metrics['accuracy']:>9.2%} "
                f"{metrics['n_samples']:>10,}  ({elapsed:.1f}s)"
            )

        print("-" * 70)
        real_nll = results["real"]["mean_nll"]
        print()
        print("Deltas (positive = real is better):")
        for name in ["zero", "shuffled"]:
            delta = results[name]["mean_nll"] - real_nll
            pct = delta / real_nll * 100
            print(f"  {name} - real = {delta:+.4f} ({pct:+.1f}%)")
    else:
        # Unaugmented: single condition (obs-only, hidden=zeros passed to forward but ignored)
        n = len(val["obs"])
        zero_hidden = np.zeros((n, args.hidden_dim), dtype=np.float32)
        t0 = time.time()
        metrics = evaluate_nll(
            model, val["obs"], val["action"], zero_hidden,
            device=args.device, batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        results["obs_only"] = metrics
        print(
            f"{'obs_only':<12} {metrics['mean_nll']:>10.4f} {metrics['accuracy']:>9.2%} "
            f"{metrics['n_samples']:>10,}  ({elapsed:.1f}s)"
        )
        print("-" * 70)

    print()
    print("=" * 70)

    # Save results
    out_path = os.path.join(os.path.dirname(args.checkpoint), "val_results.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
