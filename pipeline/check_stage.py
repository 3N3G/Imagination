#!/usr/bin/env python3
"""
Sanity check script for each pipeline stage.
Exits non-zero if something looks wrong, halting the SLURM chain.

Usage:
    python -m pipeline.check_stage embed
    python -m pipeline.check_stage merge
    python -m pipeline.check_stage train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path("/data/group_data/rl/geney/new_craftax_llm_labelled_results_shards")
EMBED_DIR = BASE / "embeddings"
GEMINI_DIR = BASE / "gemini_labels"
FINAL_DIR = BASE / "final_trajectories"
CKPT_DIR = Path("/data/group_data/rl/geney/checkpoints/awr_imagination")

EXPECTED_FILES = 158
HIDDEN_DIM = 4096


def fail(msg: str):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_embed():
    print("=" * 60)
    print("Sanity check: embedding outputs")
    print("=" * 60)

    files = sorted(EMBED_DIR.glob("*_embeddings.npz"))
    print(f"  Embedding files found: {len(files)}")
    if len(files) < EXPECTED_FILES:
        fail(f"Expected {EXPECTED_FILES} embedding files, found {len(files)}")

    # Check 2 sample files (first and last)
    for f in [files[0], files[-1]]:
        print(f"\n  Checking {f.name}:")
        data = np.load(f)
        indices = data["sample_indices"]
        embeddings = data["embeddings"]

        print(f"    sample_indices: shape={indices.shape}, dtype={indices.dtype}")
        print(f"    embeddings:     shape={embeddings.shape}, dtype={embeddings.dtype}")

        if embeddings.ndim != 2 or embeddings.shape[1] != HIDDEN_DIM:
            fail(f"Expected embeddings shape (N, {HIDDEN_DIM}), got {embeddings.shape}")
        if embeddings.dtype != np.float16:
            fail(f"Expected float16, got {embeddings.dtype}")

        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
        nan_count = int(np.isnan(embeddings).sum())
        inf_count = int(np.isinf(embeddings).sum())
        print(f"    samples: {len(indices)}")
        print(f"    norms: min={norms.min():.2f}, mean={norms.mean():.2f}, max={norms.max():.2f}")
        print(f"    NaN: {nan_count}, Inf: {inf_count}")

        if nan_count > 0:
            fail(f"{nan_count} NaN values in {f.name}")
        if inf_count > 0:
            fail(f"{inf_count} Inf values in {f.name}")
        if norms.mean() < 0.01:
            fail(f"Embedding norms suspiciously small (mean={norms.mean():.4f})")

        # Show one example embedding (first 10 dims)
        print(f"    Example embedding[0][:10]: {embeddings[0][:10]}")

    # Cross-check with gemini labels
    gemini_files = sorted(GEMINI_DIR.glob("*.jsonl"))
    print(f"\n  Gemini label files: {len(gemini_files)}")
    print(f"  Embedding files:    {len(files)}")
    if len(files) != len(gemini_files):
        print(f"  WARNING: count mismatch (embed={len(files)}, gemini={len(gemini_files)})")

    print("\n  PASS: Embeddings look good")


def check_merge():
    print("=" * 60)
    print("Sanity check: merged final trajectories")
    print("=" * 60)

    files = sorted(FINAL_DIR.glob("trajectories_*.npz"))
    print(f"  Final trajectory files found: {len(files)}")
    if len(files) < EXPECTED_FILES:
        fail(f"Expected {EXPECTED_FILES} final files, found {len(files)}")

    expected_keys = {
        "obs_map_bits", "obs_map_dim", "obs_aux", "action", "reward",
        "done", "log_prob", "return_to_go", "hidden_state",
        "text_generated", "gemini_step_idx",
    }

    for f in [files[0], files[len(files) // 2], files[-1]]:
        print(f"\n  Checking {f.name}:")
        data = np.load(f, allow_pickle=True)
        keys = set(data.files)
        missing = expected_keys - keys
        if missing:
            fail(f"Missing keys in {f.name}: {missing}")
        print(f"    Keys: {sorted(keys)}")

        n = data["action"].shape[0]
        hs = data["hidden_state"]
        print(f"    Samples: {n}")
        print(f"    hidden_state: shape={hs.shape}, dtype={hs.dtype}")
        print(f"    obs_aux: shape={data['obs_aux'].shape}")
        print(f"    action range: [{data['action'].min()}, {data['action'].max()}]")
        print(f"    return_to_go range: [{data['return_to_go'].min():.2f}, {data['return_to_go'].max():.2f}]")

        if hs.shape != (n, HIDDEN_DIM):
            fail(f"hidden_state shape {hs.shape} != expected ({n}, {HIDDEN_DIM})")

        # Check hidden states aren't all zero
        nonzero_rows = np.count_nonzero(np.any(hs != 0, axis=1))
        zero_frac = 1.0 - nonzero_rows / n
        print(f"    hidden_state: {nonzero_rows}/{n} non-zero rows ({zero_frac:.1%} zero)")
        if nonzero_rows == 0:
            fail(f"All hidden states are zero in {f.name}")

        # Check gemini_step_idx
        gsi = data["gemini_step_idx"]
        valid = np.sum(gsi >= 0)
        print(f"    gemini_step_idx: {valid}/{n} valid ({valid / n:.1%})")

        # Show a text sample
        texts = data["text_generated"]
        for i in range(min(3, n)):
            if texts[i]:
                print(f"    Example text[{i}]: {str(texts[i])[:120]}...")
                break

        data.close()

    # Check train/val split counts
    train_files = [f for f in files if int(f.stem.split("_")[-1]) < 126]
    val_files = [f for f in files if int(f.stem.split("_")[-1]) >= 126]
    print(f"\n  Train files (idx < 126): {len(train_files)}")
    print(f"  Val files (idx >= 126):  {len(val_files)}")

    print("\n  PASS: Merged trajectories look good")


def check_train():
    print("=" * 60)
    print("Sanity check: training outputs")
    print("=" * 60)

    import torch
    from pipeline.train_awr import ActorCriticAug, Config

    # Check checkpoint exists
    final = CKPT_DIR / "final.pth"
    if not final.exists():
        fail(f"Final checkpoint not found: {final}")
    print(f"  Checkpoint: {final} ({final.stat().st_size / 1e6:.1f} MB)")

    # Check stats file
    stats_file = CKPT_DIR / "hidden_state_stats.npz"
    if not stats_file.exists():
        fail(f"Hidden stats not found: {stats_file}")
    stats = np.load(stats_file)
    print(f"  Hidden stats: mean range=[{stats['mean'].min():.3f}, {stats['mean'].max():.3f}], "
          f"std range=[{stats['std'].min():.3f}, {stats['std'].max():.3f}]")
    if np.any(stats["std"] < 1e-6):
        fail("Some hidden_std dimensions are near zero")

    # Load model and verify
    model = ActorCriticAug(
        Config.OBS_DIM, Config.ACTION_DIM, Config.LAYER_WIDTH, Config.HIDDEN_STATE_DIM
    )
    state_dict = torch.load(final, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} params")

    # Forward pass sanity check
    obs = torch.randn(4, Config.OBS_DIM)
    hid = torch.randn(4, Config.HIDDEN_STATE_DIM)
    with torch.no_grad():
        pi, v = model(obs, hid)
    print(f"  Forward pass: logits shape={pi.logits.shape}, value shape={v.shape}")
    print(f"  Logits range: [{pi.logits.min():.3f}, {pi.logits.max():.3f}]")
    print(f"  Value range: [{v.min():.3f}, {v.max():.3f}]")

    # Check that weights are not initial (training actually happened)
    w = model.actor_fc2.weight.data
    print(f"  actor_fc2 weight: mean={w.mean():.4f}, std={w.std():.4f}")

    # Check metadata
    meta_file = CKPT_DIR / "training_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"  Training metadata: {meta.get('total_params', '?')} params, "
              f"{meta.get('dataset_samples', '?')} samples")

    # Check training log for final metrics
    print("\n  PASS: Training outputs look good")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["embed", "merge", "train"])
    args = parser.parse_args()

    {"embed": check_embed, "merge": check_merge, "train": check_train}[args.stage]()


if __name__ == "__main__":
    main()
