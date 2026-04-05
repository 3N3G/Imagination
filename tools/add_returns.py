import numpy as np
import glob
import os
import argparse
import concurrent.futures
import time

# ================= CONFIGURATION =================
DATA_DIR = "/data/group_data/rl/geney/craftax_labelled_results"
SEARCH_PATTERN = "trajectories_batch_*.npz"
OUTPUT_DIR = "/data/group_data/rl/geney/craftax_labelled_results_with_returns"
GAMMA = 0.99
# =================================================


def compute_return_to_go_interleaved(rewards, dones, num_envs, gamma=0.99):
    if len(rewards) % num_envs != 0:
        raise ValueError(f"Length {len(rewards)} not divisible by {num_envs}")

    rewards_mat = rewards.reshape(-1, num_envs)
    dones_mat = dones.reshape(-1, num_envs)
    returns_mat = np.zeros_like(rewards_mat, dtype=np.float32)

    T = rewards_mat.shape[0]
    next_return = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(T)):
        r = rewards_mat[t]
        d = dones_mat[t]
        current_return = r + gamma * next_return * (1.0 - d)
        returns_mat[t] = current_return
        next_return = current_return

    return returns_mat.flatten()


def process_single_file(file_info):
    fpath, num_envs, output_dir = file_info
    fname = os.path.basename(fpath)

    try:
        # Load
        with np.load(fpath, allow_pickle=True) as data:
            data_dict = {key: data[key] for key in data.files}

        if "reward" not in data_dict or "done" not in data_dict:
            return f"Skipped {fname}: missing keys"

        # Compute
        rtg = compute_return_to_go_interleaved(
            data_dict["reward"], data_dict["done"], num_envs, GAMMA
        )
        data_dict["return_to_go"] = rtg

        # Save
        if output_dir:
            save_path = os.path.join(output_dir, fname)
        else:
            save_path = fpath

        np.savez_compressed(save_path, **data_dict)
        return None

    except Exception as e:
        return f"Error {fname}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=128)

    # Auto-detect Slurm CPU allocation
    try:
        # This gets the CPUs assigned to THIS job, not the whole machine
        default_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for Mac/Windows (outside Slurm)
        default_workers = os.cpu_count() or 4

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel processes (Default: {default_workers})",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, SEARCH_PATTERN)))
    print(f"Found {len(files)} files.")
    print(f"Processing with {args.workers} workers...")

    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_args = [(f, args.num_envs, OUTPUT_DIR) for f in files]

    start_time = time.time()

    # Using ProcessPoolExecutor to bypass Python GIL for true parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_single_file, file_args))

    errors = [r for r in results if r is not None]
    print(f"Done in {time.time() - start_time:.2f}s.")

    if errors:
        print(f"Encountered {len(errors)} errors:")
        for e in errors:
            print(e)
    else:
        print("Success! All files processed.")


if __name__ == "__main__":
    main()
