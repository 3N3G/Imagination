#!/usr/bin/env python3
"""
Recompute hidden state normalization statistics using the VLM server.
This ensures the normalization stats match what the eval server produces.

Usage:
    python recompute_hidden_stats_from_server.py --server_url http://hostname:5000 --num_samples 1000
"""
import os
import argparse
import numpy as np
import glob
import requests
from tqdm import tqdm

# Prevent JAX/CUDA initialization
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


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


def load_observations(data_dir, num_samples):
    """Load observations from training data files."""
    files = sorted(glob.glob(os.path.join(data_dir, "trajectories_batch_*.npz")))
    if not files:
        raise ValueError(f"No training files found in {data_dir}")

    print(f"Found {len(files)} training files")

    # Load observations from multiple files to get diverse samples
    all_obs = []
    samples_per_file = max(1, num_samples // len(files))

    for f in files:
        if len(all_obs) >= num_samples:
            break

        try:
            data = np.load(f, mmap_mode="r")
            n = min(samples_per_file, len(data["obs"]))

            # Sample random indices for diversity
            indices = np.random.choice(len(data["obs"]), size=n, replace=False)
            obs = np.array(data["obs"][indices])
            all_obs.extend(obs)

            if len(all_obs) >= num_samples:
                break
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    all_obs = np.array(all_obs[:num_samples])
    print(f"Loaded {len(all_obs)} observations")
    return all_obs


def compute_stats(hidden_states):
    """Compute mean and std for normalization."""
    mean = np.mean(hidden_states, axis=0)
    std = np.std(hidden_states, axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero

    # Compute L2 norms
    l2_norms = np.linalg.norm(hidden_states, axis=1)
    avg_l2_norm = np.mean(l2_norms)

    return mean, std, avg_l2_norm


def main():
    parser = argparse.ArgumentParser(
        description="Recompute hidden state stats from VLM server"
    )
    parser.add_argument(
        "--server_url", type=str, default="http://localhost:5000", help="VLM server URL"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/group_data/rl/geney/craftax_labelled_results_with_returns",
        help="Training data directory (for observations)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats_server.npz",
        help="Output path for new statistics",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use (more = more accurate, but slower)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("RECOMPUTE HIDDEN STATE STATISTICS FROM VLM SERVER")
    print("=" * 70)

    # Check server is running
    print(f"\nChecking VLM server at {args.server_url}...")
    try:
        response = requests.get(f"{args.server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"  Server ready: {response.json()}")
        else:
            raise RuntimeError(f"Server returned {response.status_code}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to server: {e}")
        print("  Please start the VLM server first!")
        return

    # Load observations
    print(f"\nLoading observations from {args.data_dir}...")
    observations = load_observations(args.data_dir, args.num_samples)

    # Get hidden states from server
    print(
        f"\nGenerating hidden states from VLM server ({len(observations)} samples)..."
    )
    print("This may take a while (~1-2 seconds per sample)...")

    hidden_states = []
    for i, obs in enumerate(tqdm(observations, desc="Processing")):
        try:
            hidden = get_hidden_from_server(args.server_url, obs)
            hidden_states.append(hidden)
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            continue

    if not hidden_states:
        print("ERROR: Could not get any hidden states from server!")
        return

    hidden_states = np.array(hidden_states)
    print(
        f"\nCollected {len(hidden_states)} hidden states, shape: {hidden_states.shape}"
    )

    # Compute statistics
    print("\nComputing statistics...")
    mean, std, avg_l2_norm = compute_stats(hidden_states)

    print(f"\n--- New Statistics ---")
    print(
        f"Mean: shape={mean.shape}, range=[{mean.min():.3f}, {mean.max():.3f}], avg={mean.mean():.3f}"
    )
    print(
        f"Std:  shape={std.shape}, range=[{std.min():.3f}, {std.max():.3f}], avg={std.mean():.3f}"
    )
    print(f"Average L2 norm: {avg_l2_norm:.2f}")

    # Compare with old statistics if they exist
    old_stats_path = args.output_path.replace("_server.npz", ".npz")
    if os.path.exists(old_stats_path):
        print(f"\n--- Comparison with Old Statistics ({old_stats_path}) ---")
        old_stats = np.load(old_stats_path)
        old_mean = old_stats["mean"]
        old_std = old_stats["std"]

        print(
            f"Old Mean: range=[{old_mean.min():.3f}, {old_mean.max():.3f}], avg={old_mean.mean():.3f}"
        )
        print(
            f"New Mean: range=[{mean.min():.3f}, {mean.max():.3f}], avg={mean.mean():.3f}"
        )
        print(f"Mean Difference L2: {np.linalg.norm(mean - old_mean):.2f}")
        print()
        print(
            f"Old Std:  range=[{old_std.min():.3f}, {old_std.max():.3f}], avg={old_std.mean():.3f}"
        )
        print(
            f"New Std:  range=[{std.min():.3f}, {std.max():.3f}], avg={std.mean():.3f}"
        )
        print(f"Std Difference L2: {np.linalg.norm(std - old_std):.2f}")

        # Test normalization with new vs old stats
        test_hidden = hidden_states[:10]
        old_normalized = (test_hidden - old_mean) / old_std
        new_normalized = (test_hidden - mean) / std

        old_norm_l2 = np.linalg.norm(old_normalized, axis=1)
        new_norm_l2 = np.linalg.norm(new_normalized, axis=1)
        expected_l2 = np.sqrt(hidden_states.shape[1])

        print(f"\n--- Test Normalization (first 10 samples) ---")
        print(
            f"Expected normalized L2 (sqrt({hidden_states.shape[1]})): {expected_l2:.2f}"
        )
        print(f"With OLD stats: {old_norm_l2.mean():.2f} ± {old_norm_l2.std():.2f}")
        print(f"With NEW stats: {new_norm_l2.mean():.2f} ± {new_norm_l2.std():.2f}")

        if old_norm_l2.mean() > 100:
            print(
                f"\n⚠️  OLD stats produce normalized L2 of {old_norm_l2.mean():.2f} >> {expected_l2:.2f}"
            )
            print(f"   This confirms the distribution shift issue!")
            print(
                f"   NEW stats should fix this (normalized L2: {new_norm_l2.mean():.2f})"
            )

    # Save new statistics
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, mean=mean, std=std, avg_l2_norm=np.array(avg_l2_norm))
    print(f"\n✓ New statistics saved to: {args.output_path}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Use the new stats file for evaluation:")
    print(f"   --stats_path {args.output_path}")
    print()
    print("2. OR retrain the policy with new stats by updating awr_aug.py")
    print("   to use these server-computed stats instead of dataset stats")
    print("=" * 70)


if __name__ == "__main__":
    main()
