#!/usr/bin/env python3
"""
Diagnostic script to compare training data hidden states with VLM server outputs.
Run this with an active VLM server to understand the distribution mismatch.

Usage:
    python diagnose_hidden_state_mismatch.py --server_url http://hostname:5000 --num_samples 10
"""
import os
import argparse
import numpy as np
import glob
import requests
from PIL import Image

# Prevent JAX/CUDA initialization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def load_training_sample(data_dir, n_samples=10):
    """Load a sample of training data with hidden states."""
    files = sorted(glob.glob(os.path.join(data_dir, "trajectories_batch_*.npz")))[:1]
    if not files:
        raise ValueError(f"No training files found in {data_dir}")

    print(f"Loading from {files[0]}...")
    data = np.load(files[0], mmap_mode="r")

    # Sample random indices
    total = min(n_samples, len(data["obs"]))
    indices = np.random.choice(len(data["obs"]), size=total, replace=False)

    obs_samples = np.array(data["obs"][indices])  # (N, H, W, C)
    hidden_samples = np.array(
        data["hidden_state"][indices]
    )  # (N, 80, 2560) or (N, 2560)

    print(f"Loaded {total} samples")
    print(f"  Observation shape: {obs_samples.shape}")
    print(f"  Hidden state shape: {hidden_samples.shape}")

    return obs_samples, hidden_samples


def get_hidden_from_server(server_url, obs_np):
    """Get hidden state from VLM server."""
    # Convert to 0-1 range if needed
    if obs_np.max() > 1.0:
        obs_01 = (obs_np / 255.0).astype(np.float32)
    else:
        obs_01 = obs_np.astype(np.float32)

    response = requests.post(
        f"{server_url}/get_hidden_state", json={"obs": obs_01.tolist()}, timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Server error: {response.text}")

    data = response.json()
    return np.array(data["hidden_state"], dtype=np.float32)


def compare_distributions(train_hidden, server_hidden, stats_path=None):
    """Compare training and server hidden state distributions."""
    print("\n" + "=" * 70)
    print("=== HIDDEN STATE COMPARISON ===")
    print("=" * 70)

    # Mean pool training hidden states if needed
    if train_hidden.ndim == 3:
        train_pooled = train_hidden.mean(axis=1)  # (N, 2560)
        print(
            f"\nTraining hidden states: {train_hidden.shape} → pooled to {train_pooled.shape}"
        )
        print(f"  Tokens per sample: {train_hidden.shape[1]}")
    else:
        train_pooled = train_hidden
        print(f"\nTraining hidden states: already pooled {train_hidden.shape}")

    # Server hidden states are already pooled
    server_pooled = np.array(server_hidden)  # (N, 2560)
    print(f"Server hidden states: {server_pooled.shape}")

    # Compute raw L2 norms
    train_l2 = np.linalg.norm(train_pooled, axis=1)
    server_l2 = np.linalg.norm(server_pooled, axis=1)

    print(f"\n--- Raw L2 Norms ---")
    print(f"Training:  {train_l2.mean():.2f} ± {train_l2.std():.2f}")
    print(f"Server:    {server_l2.mean():.2f} ± {server_l2.std():.2f}")
    print(f"Ratio:     {server_l2.mean() / train_l2.mean():.3f}")

    # Per-dimension statistics
    print(f"\n--- Per-Dimension Stats (first 10 dims) ---")
    print(
        f"{'Dim':>4} {'Train Mean':>12} {'Server Mean':>12} {'Train Std':>12} {'Server Std':>12}"
    )
    for i in range(min(10, train_pooled.shape[1])):
        t_mean = train_pooled[:, i].mean()
        s_mean = server_pooled[:, i].mean()
        t_std = train_pooled[:, i].std()
        s_std = server_pooled[:, i].std()
        print(f"{i:>4} {t_mean:>12.4f} {s_mean:>12.4f} {t_std:>12.4f} {s_std:>12.4f}")

    # Load normalization stats if provided
    if stats_path and os.path.exists(stats_path):
        print(f"\n--- Using Normalization Stats from {stats_path} ---")
        stats = np.load(stats_path)
        norm_mean = stats["mean"]
        norm_std = stats["std"]

        # Normalize both
        train_normalized = (train_pooled - norm_mean) / norm_std
        server_normalized = (server_pooled - norm_mean) / norm_std

        train_norm_l2 = np.linalg.norm(train_normalized, axis=1)
        server_norm_l2 = np.linalg.norm(server_normalized, axis=1)

        expected_l2 = np.sqrt(train_pooled.shape[1])

        print(f"\n--- Normalized L2 Norms ---")
        print(f"Expected (sqrt({train_pooled.shape[1]})): {expected_l2:.2f}")
        print(f"Training:  {train_norm_l2.mean():.2f} ± {train_norm_l2.std():.2f}")
        print(f"Server:    {server_norm_l2.mean():.2f} ± {server_norm_l2.std():.2f}")
        print(f"Ratio:     {server_norm_l2.mean() / train_norm_l2.mean():.3f}")

        if server_norm_l2.mean() > 100:
            print(f"\n⚠️  MAJOR DISTRIBUTION SHIFT DETECTED!")
            print(
                f"   Server normalized L2 ({server_norm_l2.mean():.2f}) >> Expected ({expected_l2:.2f})"
            )
            print(f"   This explains why the policy performs badly during eval.")

    return train_pooled, server_pooled


def main():
    parser = argparse.ArgumentParser(description="Diagnose hidden state mismatch")
    parser.add_argument(
        "--server_url", type=str, default="http://localhost:5000", help="VLM server URL"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/group_data/rl/geney/craftax_labelled_results_with_returns",
        help="Training data directory",
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz",
        help="Path to normalization statistics",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to compare"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("HIDDEN STATE MISMATCH DIAGNOSTIC")
    print("=" * 70)

    # Check server is running
    print(f"\nChecking VLM server at {args.server_url}...")
    try:
        response = requests.get(f"{args.server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"  Server ready: {response.json()}")
        else:
            print(f"  Warning: Server returned {response.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to server: {e}")
        print("  Please start the VLM server first!")
        return

    # Load training data
    obs_samples, train_hidden = load_training_sample(args.data_dir, args.num_samples)

    # Get hidden states from server for same observations
    print(f"\nGetting hidden states from VLM server...")
    server_hidden = []
    for i, obs in enumerate(obs_samples):
        print(f"  Processing {i+1}/{len(obs_samples)}...", end="\r")
        try:
            hidden = get_hidden_from_server(args.server_url, obs)
            server_hidden.append(hidden)
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue
    print()

    if not server_hidden:
        print("ERROR: Could not get any hidden states from server!")
        return

    server_hidden = np.array(server_hidden)
    print(f"Got {len(server_hidden)} hidden states from server")

    # Compare distributions
    compare_distributions(
        train_hidden[: len(server_hidden)], server_hidden, args.stats_path
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
