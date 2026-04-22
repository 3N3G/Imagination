"""Build a bitpacked-stage top-K row-count subset.

Analogue of build_psf_size_subsets.py, but operates on the post-filter_and_repack
stage (no hidden_state / no text_generated yet). Output directory has the same
npz schema as the input so gemini_label.py can consume it directly.

Usage:
  PYTHONPATH=. python -m pipeline.build_bitpacked_top_subset \\
      --input-dir /path/to/filtered_trajectories \\
      --output-dir /path/to/filtered_trajectories_psf_v2_top2M \\
      --target-rows 2000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# Keep all bitpacked fields plus auxiliary columns; no hidden_state here.
KEYS_KEEP = [
    "obs_map_bits",
    "obs_aux",
    "action",
    "reward",
    "done",
    "log_prob",
    "return_to_go",
]
SCALAR_KEYS = ["obs_map_dim"]


def find_episodes(done: np.ndarray):
    done_idx = np.where(done)[0]
    eps = []
    start = 0
    for di in done_idx:
        eps.append((start, di + 1))
        start = di + 1
    return eps


def scan_episodes(files):
    t0 = time.time()
    meta = []
    total_rows = 0
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        done = np.asarray(d["done"]).reshape(-1).astype(bool)
        rtg = np.asarray(d["return_to_go"]).reshape(-1).astype(np.float32)
        total_rows += len(done)
        d.close()
        for s, e in find_episodes(done):
            meta.append((fi, s, e, float(rtg[s])))
        if (fi + 1) % 20 == 0 or fi == 0:
            print(f"  scanned {fi + 1}/{len(files)}   "
                  f"episodes={len(meta)}   rows={total_rows}   "
                  f"elapsed={time.time() - t0:.1f}s")
    print(f"  done scan: {len(meta)} episodes, {total_rows} rows, "
          f"{time.time() - t0:.1f}s")
    return meta, total_rows


def pick_subset(meta_sorted, target_rows):
    picked = []
    rows = 0
    for entry in meta_sorted:
        fi, s, e, r = entry
        picked.append(entry)
        rows += (e - s)
        if rows >= target_rows:
            break
    return picked, rows


def write_subset(picked, files, out_dir: Path, per_file_rows: int = 100_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_file: dict[int, list[tuple[int, int]]] = {}
    for fi, s, e, _ in picked:
        by_file.setdefault(fi, []).append((s, e))

    sample = np.load(files[0], allow_pickle=False)
    scalars = {k: sample[k].copy() for k in SCALAR_KEYS if k in sample.files}
    sample.close()

    buf: dict[str, list[np.ndarray]] = {k: [] for k in KEYS_KEEP}
    buf_rows = 0
    out_idx = 0
    total_written = 0

    def flush():
        nonlocal buf, buf_rows, out_idx, total_written
        if buf_rows == 0:
            return
        payload = {k: np.concatenate(buf[k], axis=0) for k in KEYS_KEEP}
        payload.update(scalars)
        out_path = out_dir / f"trajectories_{out_idx:06d}.npz"
        np.savez(out_path, **payload)
        total_written += buf_rows
        print(f"    wrote {out_path.name}  rows={buf_rows}  cum={total_written}")
        buf = {k: [] for k in KEYS_KEEP}
        buf_rows = 0
        out_idx += 1

    for fi in sorted(by_file.keys()):
        d = np.load(files[fi], allow_pickle=False)
        arrs = {k: d[k] for k in KEYS_KEEP}
        d.close()
        for s, e in by_file[fi]:
            for k in KEYS_KEEP:
                buf[k].append(arrs[k][s:e])
            buf_rows += (e - s)
            if buf_rows >= per_file_rows:
                flush()
    flush()
    print(f"  finished {out_dir.name}  total_rows={total_written}")
    return total_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Bitpacked stage dir (e.g. filtered_trajectories)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-rows", type=int, required=True)
    p.add_argument("--per-file-rows", type=int, default=100_000)
    args = p.parse_args()

    files = sorted(args.input_dir.glob("trajectories_*.npz"))
    if not files:
        raise SystemExit(f"No trajectories_*.npz in {args.input_dir}")
    print(f"Input: {args.input_dir}  files={len(files)}")
    print(f"Output: {args.output_dir}")
    print(f"Target rows: {args.target_rows}")

    print("\n--- Phase 1: scan episodes ---")
    meta, total_rows = scan_episodes(files)
    meta_sorted = sorted(meta, key=lambda x: x[3], reverse=True)
    returns = np.array([m[3] for m in meta_sorted], dtype=np.float32)
    print(f"  return dist: min={returns.min():.2f} max={returns.max():.2f} "
          f"mean={returns.mean():.2f} median={float(np.median(returns)):.2f}")

    print(f"\n--- Phase 2: pick + write ---")
    picked, actual = pick_subset(meta_sorted, args.target_rows)
    kept_returns = np.array([e[3] for e in picked], dtype=np.float32)
    print(f"  picked {len(picked)} episodes, target={args.target_rows} "
          f"actual={actual}   return range kept: "
          f"[{kept_returns.min():.2f}, {kept_returns.max():.2f}]")
    written = write_subset(picked, files, args.output_dir,
                           per_file_rows=args.per_file_rows)

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_rows": args.target_rows,
        "actual_rows": int(written),
        "n_episodes": len(picked),
        "total_scanned_rows": int(total_rows),
        "total_scanned_episodes": len(meta),
        "return_min_kept": float(kept_returns.min()),
        "return_max_kept": float(kept_returns.max()),
    }
    manifest_path = args.output_dir / "subset_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
