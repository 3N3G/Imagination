#!/usr/bin/env python
"""
Post-processing script to add text observations to saved trajectory NPZ files.

This runs AFTER ppo.py data collection completes, adding text_obs to each file
using obs_to_text to decode the symbolic observation back to text format.

Usage:
    python add_text_obs.py --input_dir /path/to/npz/files --num_workers 32

The script will:
1. Find all NPZ files without text_obs
2. For each file, decode observations to text using obs_to_text
3. Save updated NPZ files with text_obs included
"""

import os
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from tqdm import tqdm

# Import the decoder
from obs_to_text import obs_to_text, obs_to_text_batch, TOTAL_OBS_SIZE


def process_single_file(npz_path: str) -> tuple[str, bool, str]:
    """
    Process a single NPZ file to add text observations.
    
    Args:
        npz_path: Path to the NPZ file
        
    Returns:
        (filepath, success, message)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Check if already processed
        if "text_obs" in data.files:
            return npz_path, True, "Already has text_obs"
        
        # Get observations
        obs = data["obs"]
        num_samples = len(obs)
        
        # Verify observation size
        if obs.shape[1] != TOTAL_OBS_SIZE:
            return npz_path, False, f"Wrong obs size: {obs.shape[1]} != {TOTAL_OBS_SIZE}"
        
        # Generate text for each observation using the decoder
        text_obs_list = obs_to_text_batch(obs)
        
        # Create new data dict with all original data plus text_obs
        new_data = {key: data[key] for key in data.files}
        new_data["text_obs"] = np.array(text_obs_list, dtype=object)
        
        # Save back (overwrite)
        np.savez_compressed(npz_path, **new_data)
        
        return npz_path, True, f"Added text_obs ({num_samples} samples)"
        
    except Exception as e:
        import traceback
        return npz_path, False, f"Error: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Add text observations to trajectory NPZ files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with NPZ files")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for files")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Find all NPZ files
    npz_files = sorted(glob.glob(str(input_dir / args.pattern)))
    print(f"Found {len(npz_files)} NPZ files in {input_dir}")
    
    if not npz_files:
        return
    
    # Process files
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # Sequential processing (safe, predictable)
    for npz_path in tqdm(npz_files, desc="Processing"):
        filepath, success, message = process_single_file(npz_path)
        
        if "Already has text_obs" in message:
            skip_count += 1
        elif success:
            success_count += 1
            if success_count <= 3:
                print(f"  {Path(filepath).name}: {message}")
        else:
            fail_count += 1
            print(f"  FAILED {Path(filepath).name}: {message}")
    
    print(f"\n{'='*50}")
    print(f"Complete: {success_count} processed, {skip_count} skipped, {fail_count} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

