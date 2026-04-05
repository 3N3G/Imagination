import numpy as np
import glob
import os
import tqdm

# ================= CONFIGURATION =================
DATA_DIR = "/data/group_data/rl/craftax_labelled_results_with_returns"
DATA_GLOB = "trajectories_batch_*.npz"
NUM_ENVS = 128  # Must match your generation config
# =================================================


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, DATA_GLOB)))
    print(f"Found {len(files)} files.")

    total_episode_rewards = []

    # We maintain a running sum for each of the 128 environments
    # This persists across files!
    running_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    total_tuples = 0

    for fpath in tqdm.tqdm(files, desc="Scanning Dataset"):
        try:
            with np.load(fpath) as data:
                # Shape: (Time * Envs,)
                rewards = data["reward"]
                dones = data["done"]

                # Reshape to (Time, Envs)
                # If the file size isn't perfectly divisible, truncate the remainder
                # (Standard safety for interleaved data)
                n_steps = len(rewards) // NUM_ENVS
                rewards = rewards[: n_steps * NUM_ENVS].reshape(n_steps, NUM_ENVS)
                dones = dones[: n_steps * NUM_ENVS].reshape(n_steps, NUM_ENVS)

                # Iterate through time in this batch
                for t in range(n_steps):
                    r_t = rewards[t]  # (128,)
                    d_t = dones[t]  # (128,)

                    # 1. Add reward to current running total
                    running_rewards += r_t
                    total_tuples += NUM_ENVS

                    # 2. If episode ended, record the total and reset
                    # d_t is 1.0 (True) or 0.0 (False)
                    mask = d_t > 0.5

                    if np.any(mask):
                        # Extract the totals for finished episodes
                        finished_scores = running_rewards[mask]
                        total_episode_rewards.extend(finished_scores.tolist())

                        # Reset the running totals for those specific envs
                        running_rewards[mask] = 0.0

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if len(total_episode_rewards) == 0:
        print("No completed episodes found in dataset.")
        return

    stats = np.array(total_episode_rewards)

    print(f"\n=== DATASET TOTAL REWARD STATISTICS ===")
    print(f"Total Episodes Counted: {len(stats)}")
    print(f"Total Tuples Counted:   {total_tuples}")
    print(f"Mean Total Reward:      {np.mean(stats):.4f}")
    print(f"Median Total Reward:    {np.median(stats):.4f}")
    print(f"Std Dev:                {np.std(stats):.4f}")
    print(f"Min:                    {np.min(stats):.4f}")
    print(f"Max:                    {np.max(stats):.4f}")


if __name__ == "__main__":
    main()
