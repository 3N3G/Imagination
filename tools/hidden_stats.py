import torch
import numpy as np
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def load_file(filepath):
    """
    CPU-ONLY worker. Do NOT initialize torch or cuda here.
    """
    try:
        # np.load is CPU-bound (decompression)
        with np.load(filepath) as data:
            if "hidden_state" not in data:
                return None
            raw = data["hidden_state"]
            # Mean pool if (N, S, D) -> (N, D)
            return raw.mean(axis=1) if raw.ndim == 3 else raw
    except Exception:
        return None


def compute_all_stats(data_dir, output_path, num_files=None):
    files = sorted(glob.glob(str(Path(data_dir) / "trajectories_batch_*.npz")))
    if num_files:
        files = files[:num_files]

    print(f"Found {len(files)} files. Initializing GPU...")

    # Initialize CUDA ONLY in the main process
    device = torch.device("cuda")

    sum_x = None
    sum_x2 = None
    sum_l2 = 0.0
    total_samples = 0

    # Use imap to stream data from CPU workers to GPU
    # Setting chunksize helps efficiency with many small files
    with Pool(cpu_count()) as pool:
        iterator = pool.imap_unordered(load_file, files, chunksize=4)

        for arr in tqdm(iterator, total=len(files), desc="Computing Stats"):
            if arr is None:
                continue

            # Move to GPU for math
            batch_data = torch.from_numpy(arr).to(device, dtype=torch.float32)

            if sum_x is None:
                dim = batch_data.shape[1]
                sum_x = torch.zeros(dim, device=device)
                sum_x2 = torch.zeros(dim, device=device)

            # High-speed GPU reductions
            sum_x += batch_data.sum(dim=0)
            sum_x2 += (batch_data**2).sum(dim=0)
            sum_l2 += torch.norm(batch_data, p=2, dim=1).sum().item()
            total_samples += batch_data.size(0)

    # Final calculations on GPU
    mean = sum_x / total_samples
    var = (sum_x2 / total_samples) - (mean**2)
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    avg_l2_norm = sum_l2 / total_samples

    # Save
    np.savez(
        output_path,
        mean=mean.cpu().numpy(),
        std=std.cpu().numpy(),
        # Wrap scalars in np.array
        avg_l2_norm=np.array(avg_l2_norm),
        total_samples=np.array(total_samples),
    )

    print(f"\nSuccess! Total Samples: {total_samples}")
    print(f"Avg L2 Norm: {avg_l2_norm:.4f}")


if __name__ == "__main__":
    DATA_DIR = "/data/group_data/rl/geney/craftax_labelled_results_with_returns"
    OUT_PATH = (
        "/data/group_data/rl/geney/checkpoints/awr_augmented/hidden_state_stats.npz"
    )
    compute_all_stats(DATA_DIR, OUT_PATH)
